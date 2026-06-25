#!/usr/bin/env python3
"""
generate_instructions.py
Generates synthetic instruction-following data in Hinglish, Hindi, and English
using NVIDIA Nemotron-Super via the NIM API (OpenAI-compatible endpoint).

Usage:
    export NVIDIA_API_KEY="nvapi-..."
    python data/generate_instructions.py --count 15000 --output data/raw_instructions.jsonl

Output format:
    {"instruction": "...", "output": "..."}
    {"system": "...", "instruction": "...", "output": "..."}
"""

import os
import json
import argparse
import time
import concurrent.futures
from openai import OpenAI

# Load .env if present
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))

# ─── Seed instructions for each language ─────────────────────────────────────

SEED_TASKS = {
    "translation": {
        "prompt": "Generate a {lang} instruction asking to translate text {from_lang} to {to_lang}, then provide the correct translation.",
        "count": 0
    },
    "summarization": {
        "prompt": "Generate a {lang} instruction asking to summarize a paragraph, then provide a summary.",
        "count": 0
    },
    "qa": {
        "prompt": "Generate a {lang} question-answer pair about {topic}.",
        "count": 0
    },
    "brainstorming": {
        "prompt": "Generate a {lang} instruction asking for ideas about {topic}, then provide 3-5 ideas.",
        "count": 0
    },
    "classification": {
        "prompt": "Generate a {lang} instruction asking to classify something, then provide the classification.",
        "count": 0
    },
    "creative": {
        "prompt": "Generate a {lang} creative writing instruction, then provide a short story or poem.",
        "count": 0
    },
    "grammar": {
        "prompt": "Generate a {lang} instruction asking to correct grammar in a sentence, then provide the corrected version.",
        "count": 0
    },
}


def build_client():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY not set. Get one at https://build.nvidia.com/nim"
        )
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def fetch_single_record(client, model: str, seed: str, temperature: float) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dataset generator. Generate one instruction-output pair "
                        "in the specified language and script. "
                        "Output ONLY valid JSON with keys 'instruction' and 'output'. "
                        "The 'output' value MUST be a plain string (text only), not a JSON object or array. "
                        "Use Hinglish (Hindi+English mix in Roman script), Hindi (Devanagari), "
                        "or English as requested."
                    ),
                },
                {"role": "user", "content": seed},
            ],
            temperature=temperature,
            max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()
        record = try_parse_json(text)
        if record and "instruction" in record and "output" in record:
            return record
        else:
            print(f"  [warn] Failed to parse: {text[:80]}...")
            return None
    except Exception as e:
        print(f"  [error] {e}")
        time.sleep(2)
        return None


def generate_batch(client, model: str, seed_prompts: list[str], temperature: float = 0.8, workers: int = 10) -> list[dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_single_record, client, model, seed, temperature): seed for seed in seed_prompts}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            if record:
                results.append(record)
    return results


def try_parse_json(text: str) -> dict | None:
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def build_seed_prompts(count: int, languages: list[str], scripts: list[str]) -> list[str]:
    topics = [
        "machine learning", "artificial intelligence", "cloud computing",
        "climate change", "Indian economy", "cricket", "Bollywood",
        "Indian festivals", "technology", "education",
        "healthcare in India", "startups", "remote work", "data science",
    ]
    combos = []
    for lang in languages:
        for script in scripts:
            if lang == "hi" and script == "roman":
                continue
            combos.append((lang, script))

    prompts = []
    task_types = list(SEED_TASKS.keys())
    for i in range(count):
        lang, script = combos[i % len(combos)]
        task = task_types[i % len(task_types)]
        topic = topics[i % len(topics)]
        lang_label = {"hi": "Hindi (Devanagari)", "en": "English", "hinglish": "Hinglish (Roman)"}.get(lang, lang)
        prompt = (
            f"Generate a {lang_label} instruction-output pair. "
            f"Task type: {task}. "
            f"Topic: {topic}. "
            f"Make the instruction realistic and natural. "
            f"The 'output' field must be a plain text string (NOT a JSON object or list). "
            f"Output as JSON with keys 'instruction' and 'output'."
        )
        prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate instruction data via NVIDIA NIM")
    parser.add_argument("--count", type=int, default=1000, help="Number of records to generate")
    parser.add_argument("--output", type=str, default="data/raw_instructions.jsonl", help="Output JSONL path")
    parser.add_argument("--languages", nargs="+", default=["hinglish", "hi", "en"],
                        choices=["hinglish", "hi", "en"], help="Languages to generate")
    parser.add_argument("--scripts", nargs="+", default=["roman", "devanagari"],
                        choices=["roman", "devanagari"], help="Scripts to use")
    parser.add_argument("--model", type=str, default="nvidia/llama-3.3-nemotron-super-49b-v1",
                        help="NVIDIA NIM model ID")
    parser.add_argument("--batch_size", type=int, default=10, help="Records per batch")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent threads for API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file, appending new records")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Count existing records if resuming
    existing_count = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_count += 1
        if existing_count >= args.count:
            print(f"Already have {existing_count}/{args.count} records. Nothing to do.")
            return
        print(f"Resuming: {existing_count} records exist, generating {args.count - existing_count} more")

    client = build_client()

    seed_prompts = build_seed_prompts(args.count, args.languages, args.scripts)

    # Skip seeds that already have records (resume mode)
    if existing_count > 0:
        seed_prompts = seed_prompts[existing_count:]

    print(f"Generating up to {args.count} records (batch_size={args.batch_size}, workers={args.workers})...")
    all_records = []
    with open(args.output, "a", encoding="utf-8") as f:
        for i in range(0, len(seed_prompts), args.batch_size):
            batch = seed_prompts[i : i + args.batch_size]
            records = generate_batch(client, args.model, batch, args.temperature, args.workers)
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            all_records.extend(records)
            total = existing_count + len(all_records)
            print(f"  Progress: {total}/{args.count} records")
            if total >= args.count:
                break

    total = existing_count + len(all_records)
    print(f"\nDone! {total} records saved to {args.output}")
    lang_counts = {}
    for rec in all_records:
        inst = rec.get("instruction", "")
        if any("\u0900" <= c <= "\u097F" for c in inst):
            tag = "hindi"
        elif any(c in inst for c in "ko hai se mein ka ki"):
            tag = "hinglish"
        else:
            tag = "english"
        lang_counts[tag] = lang_counts.get(tag, 0) + 1
    print(f"Language breakdown (this run): {lang_counts}")


if __name__ == "__main__":
    main()
