# Async LLM Dataset Generation: Patterns & Implementation Notes

> This document explains the engineering decisions behind `data/generate_instructions.py` and `track_progress.sh` — a production-grade approach to running a long-running, parallel LLM API task reliably under real-world constraints (rate limits, transient errors, limited local compute).

---

## The Problem

Generating 15,000 synthetic instruction-output pairs via an LLM API is not a simple loop. In practice, you hit:

- **Rate limits (429)** — per-key request quotas exhausted by concurrent workers
- **Transient server errors (500, 502)** — upstream inference failures on the provider side
- **Bad model outputs** — the model ignores your JSON schema and wraps output in prose
- **Interruptions** — process crashes, machine sleeps, Ctrl+C, server restarts

Naively running this in a single thread would take days. Running it with high concurrency without any resilience destroys all your API quota on retries that go nowhere.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              generate_instructions.py            │
│                                                  │
│  ┌──────────┐   ┌──────────────────────────────┐ │
│  │  Resume  │   │   ThreadPoolExecutor          │ │
│  │  Logic   │──▶│   (10 workers)                │ │
│  │ (--resume│   │                               │ │
│  │  flag)   │   │  Worker 0 ──▶ Key 0           │ │
│  └──────────┘   │  Worker 1 ──▶ Key 1           │ │
│                 │  Worker 2 ──▶ Key 2           │ │
│  ┌──────────┐   │  Worker 3 ──▶ Key 3           │ │
│  │  6 API   │   │  ...                          │ │
│  │  Keys    │   │  Worker 9 ──▶ Key 3 (round-  │ │
│  │ (comma-  │   │               robin)          │ │
│  │separated │   └──────────────────────────────┘ │
│  │ in .env) │              │                      │
│  └──────────┘              ▼                      │
│                   fetch_single_record()           │
│                   (key rotation + backoff)        │
│                              │                   │
│                              ▼                   │
│                    data/raw_instructions.jsonl   │
│                    (append-mode, flushed each    │
│                     record — crash-safe)         │
└─────────────────────────────────────────────────┘
```

---

## Key Patterns

### 1. Multi-Key Round-Robin Client Pool

Instead of a single API client, we build a **pool of clients** — one per API key — loaded from a comma-separated `NVIDIA_API_KEY` environment variable:

```python
def build_clients():
    keys = [k.strip() for k in os.environ["NVIDIA_API_KEY"].split(",") if k.strip()]
    return [OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=k) for k in keys]
```

Workers are assigned a `start_idx` so each starts on a different key, spreading load evenly:

```python
executor.submit(fetch_single_record, clients, model, seed, temperature, i % len(clients))
```

**Why this matters:** With a single key and 10 workers, every worker competes for the same quota. With 6 keys, each worker has a dedicated starting key — 6× the effective throughput before rate limiting kicks in.

---

### 2. Immediate Key Rotation on 429 (No Sleep-on-Same-Key)

The most important fix. The naive approach is to sleep exponentially on the *same* key after a 429. This is catastrophically slow because:
- All workers sleep simultaneously
- They all retry the same exhausted key at the same time
- You get a "thundering herd" effect where they all hit 429 again

Our approach: **on a 429, immediately try the next key** in the pool without sleeping:

```python
for round_ in range(max_rounds):       # up to 3 full sweeps
    for key_offset in range(n):        # try each key in turn
        client = clients[(start_idx + key_offset) % n]
        try:
            # ... make API call ...
        except Exception as e:
            if "429" in str(e):
                if key_offset < n - 1:
                    continue           # ✅ try next key immediately
                else:
                    # All keys exhausted — now sleep before next round
                    sleep_time = base_sleep * (2 ** round_) + random.uniform(0, 1)
                    time.sleep(sleep_time)
```

Only after **all 6 keys** have been tried does the worker sleep — and even then for only 1–2 seconds. This transforms the failure mode from "everyone sleeps for 16s" to "rotate quickly, sleep only if truly exhausted."

---

### 3. Differentiated Error Handling

Not all errors are equal. We handle them differently:

| Error | Strategy | Reason |
|---|---|---|
| `429 Too Many Requests` | Rotate to next key immediately | Quota issue, different key may have quota |
| `500 Already Borrowed` | Sleep 2s, retry same key | Transient server bug, not quota-related |
| `502 Bad Gateway` | Sleep 2s, retry same key | Transient infra blip |
| Any other exception | Log and drop record | Unknown — don't retry indefinitely |
| Parse failure | Drop record | Model output malformed — retrying won't help |

---

### 4. Crash-Safe Resumable Writes

Records are written **one at a time** in append mode with `f.flush()` after each write:

```python
with open(args.output, "a", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()   # ← ensures disk write even if process dies
```

Combined with `--resume`, which counts existing lines and skips that many seed prompts, **the script can be interrupted and restarted at any time** without losing a single completed record or re-generating already-saved ones.

---

### 5. Live Progress Tracking (`track_progress.sh`)

A small bash script that maintains a **state file** (`/tmp/sarvam_gen_tracker.state`) between runs to compute a real rate:

```bash
current=$(wc -l < data/raw_instructions.jsonl)
prev_count=$(cut -d',' -f1 "$STATE_FILE")
prev_time=$(cut -d',' -f2 "$STATE_FILE")

delta=$(( current - prev_count ))
elapsed=$(( now - prev_time ))
rate=$(python3 -c "print(round($delta / ($elapsed / 60), 1))")
```

On each 5-minute cron tick, it computes:
- **Records/min** — delta since last check divided by elapsed minutes
- **ETA** — remaining records divided by current rate

This is triggered automatically by an Antigravity scheduled cron every 5 minutes, reporting to the conversation in real time.

---

## What We Learned

1. **Rate limit rotation > exponential backoff** — when you have multiple keys, rotating is always better than sleeping on an exhausted key.
2. **Flush every record** — with long-running jobs, you can't afford to lose records to a crash. `f.flush()` costs almost nothing.
3. **Resume from file count, not memory** — using `wc -l` on the output file as the source of truth means any restart (crash, Ctrl+C, reboot) picks up cleanly.
4. **Separate error types** — a `429` and a `500` require completely different responses. Treating them the same is a common mistake.
5. **State file tracking** — for measuring throughput of a long-running job, a simple two-field state file (count, timestamp) is all you need.

---

## Files

| File | Purpose |
|---|---|
| `data/generate_instructions.py` | Main generation script with all resilience patterns |
| `track_progress.sh` | Bash progress tracker with rate + ETA calculation |
| `.env` | Comma-separated `NVIDIA_API_KEY` for multi-key pool |
