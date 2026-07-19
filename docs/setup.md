# Indic Instructor — setup

Status: [STATUS.md](./STATUS.md). Training notes: [training_unsloth.md](./training_unsloth.md). Scope: [scope.md](./scope.md). Root README for full pipeline.

## Prerequisites

- Python env (project `requirements.txt` / Colab `requirements-colab.txt`)  
- NVIDIA NIM (or OpenAI-compatible) key for **teacher** data generation  
- CUDA GPU for real Sarvam LoRA; CPU dry-run works without  

## Typical flow

```bash
cd "~/Projects/Fine-tuning Sarvam-1"
# env: NVIDIA_API_KEY / HF tokens as needed — never commit
make generate    # or python scripts path per Makefile
make split
make train-dry   # smoke
# make train     # GPU
make serve
```

## Hygiene

- Ignore large logs / local NIM dumps (see `.gitignore`)  
- Do not commit model weights if repo policy excludes them  
