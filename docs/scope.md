# Indic Instructor — Project Scope

This repository is **only** about fine-tuning, evaluating, serving, and caching a Sarvam-1 instruction model for Hinglish / Hindi / English.

## In scope

| Area | Status / direction |
|------|--------------------|
| Synthetic instruction data (NIM / Nemotron) | Implemented |
| Train/val split | Implemented |
| Unsloth LoRA SFT on Sarvam-1 | Implemented (GPU); dry-run on CPU |
| BLEU-1 / ROUGE-L / latency eval | Implemented |
| Base vs fine-tuned comparison charts | Implemented |
| FastAPI inference (vLLM or HF) | Implemented |
| LoRA adapter load at serve time | Implemented (HF path) |
| In-memory LRU response cache | Implemented (`/generate`) |
| Merged 16-bit export for vLLM | Train path can export |

## Out of scope

- Unrelated product apps or multi-agent systems
- Shared naming, folders, or modules with other personal projects

## Naming conventions

| Thing | Name |
|-------|------|
| Project / adapter | `sarvam-1-indic-instructor` |
| W&B project | `sarvam-1-indic-instructor` |
| Merged export | `models/sarvam-1-indic-instructor-merged-16bit` |
| API title | Indic Instructor API |

## Future work (this repo only)

1. Finish ~15k instruction dataset and train a real adapter on GPU/Colab
2. Measure and publish real BLEU/ROUGE numbers
3. Optional Redis / disk cache for multi-worker serving
4. Prefix / KV cache experiments under vLLM
5. Verify ChatML vs Sarvam-1 native chat template
6. Quantized serve (AWQ/GPTQ) once quality is locked
