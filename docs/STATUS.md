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
| Synthetic data (Nemotron-Super 49B teacher) | ✅ Generator ready (`make generate`, `--resume`) |
| Sample data on disk | Partial (~hundreds; target ~15k) |
| Train/val split | `make split` after generation |
| Real Sarvam-1 LoRA on GPU | Not checked in — train on Colab/CUDA |
| CPU dry-run | ✅ `make train-dry`, `make eval-dry` |
| Serving + cache | ✅ `make serve` |

---

## Next

- [ ] Scale data gen toward target pair count  
- [ ] Run full Unsloth LoRA on GPU; commit or Hub-push adapters  
- [ ] Publish eval numbers (BLEU-1, ROUGE-L, latency) for best checkpoint  

---

## Resume

```bash
cd "~/Projects/Fine-tuning Sarvam-1"
# see README + docs/setup.md + docs/scope.md + docs/training_unsloth.md
make help   # if defined
```

Scope only: data → train → eval → serve. No product apps beyond the model API.
