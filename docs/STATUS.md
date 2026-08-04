# Indic Instructor (Sarvam-1) — status handoff

| Field | Value |
|-------|--------|
| **As of** | 2026-07-19 |
| **Branch** | `master` |
| **Repo** | `sarvam-1-indic-instructor` |
| **Path** | `~/Projects/Fine-tuning Sarvam-1` |
| **Product** | Fine-tune Sarvam-1 (2B) for Hinglish / Hindi / English instruction following |

---

## Pipeline status

| Step | State |
|------|--------|
| Synthetic data (Nemotron-Super 49B teacher) | ✅ Completed (15,017 records in `data/raw_instructions.jsonl`) |
| Sample data on disk | ✅ 15,017 records on disk & committed |
| Train/val split | ✅ Completed (`data/train.jsonl`: 13,515 / `data/val.jsonl`: 1,502) |
| Real Sarvam-1 LoRA on GPU | ⏳ Next step — Train on Google Colab GPU |
| CPU dry-run | ✅ Verified (`make train-dry`, `make eval-dry`) |
| Serving + cache | ✅ Verified (`make serve`) |

---

## Next

- [x] Scale data gen toward target pair count (15,017 records complete)  
- [ ] Run full Unsloth LoRA on Colab GPU (`make train-colab`); save adapters  
- [ ] Evaluate benchmark numbers (BLEU-1, ROUGE-L, latency, tok/s)  
- [ ] Serve fine-tuned model via FastAPI / vLLM  

---

## Resume

```bash
cd "~/Projects/Fine-tuning Sarvam-1"
# see README + docs/setup.md + docs/scope.md + docs/training_unsloth.md
make help   # if defined
```

Scope only: data → train → eval → serve. No product apps beyond the model API.
