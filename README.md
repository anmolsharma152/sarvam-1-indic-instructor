# Indic Instructor

**Instruction-tuned Sarvam-1 for Hinglish, Hindi & English**

A fine-tuned `sarvamai/sarvam-1` (3B) model that follows instructions in Hinglish, Hindi, and English across both Roman and Devanagari scripts. Built for the Indic language community — no other 3B instruction model exists at this scale for code-switched Hinglish.

---

## What It Does

| Language | Script | Example |
|----------|--------|---------|
| English | Roman | "Summarize this job posting in 2 sentences." |
| Hindi | Devanagari | "इस जॉब पोस्टिंग को दो वाक्यों में समझाओ।" |
| Hinglish | Roman | "Is job posting ko 2 sentences mein summarize karo." |

Tasks covered: summarization, translation, Q&A, brainstorming, classification, creative writing, grammar correction.

---

## Dataset

The training data (`data/train.jsonl`, `data/val.jsonl`) is generated synthetically using NVIDIA Nemotron-Super via the NVIDIA NIM API. Each record follows:

```json
{
  "instruction": "Translate to Hinglish: 'I am going to the market'",
  "output": "Main market ja raha hoon"
}
```

Optionally with a system prompt:

```json
{
  "system": "You are a helpful assistant. Always respond in Hinglish.",
  "instruction": "What is machine learning?",
  "output": "Machine learning ek aisi technique hai jahan computer data se seekhta hai..."
}
```

### Dataset Generation

```bash
export NVIDIA_API_KEY="your_key"
python data/generate_instructions.py \
    --count 15000 \
    --output data/raw_instructions.jsonl \
    --languages hi en hinglish \
    --scripts devanagari roman
```

Then split into train/val:

```bash
python data/split.py --input data/raw_instructions.jsonl --train data/train.jsonl --val data/val.jsonl
```

---

## Training

### Colab (T4 GPU)

```python
# Clone and setup
!git clone https://github.com/anmolsharma152/Fine-tuning-on-Job-Description-Corpus.git
%cd Fine-tuning-on-Job-Description-Corpus
!bash setup/colab_setup.sh

# Restart runtime → then:
!WANDB_MODE="offline" python training/train.py \
    --model_id sarvamai/sarvam-1 \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir models/sarvam-1-indic-instructor \
    --epochs 3 \
    --batch_size 4
```

### Local CPU Dry-Run

```bash
python training/train.py --dry-run
```

---

## Evaluation

```bash
# On fine-tuned model
python eval/benchmark.py \
    --model sarvamai/sarvam-1 \
    --adapter models/sarvam-1-indic-instructor \
    --test_file data/val.jsonl \
    --num_samples 500 \
    --output eval/results.json
```

### Metrics

| Metric | Method |
|--------|--------|
| BLEU-4 | N-gram overlap on held-out translations |
| ROUGE-L | Longest common subsequence for summarization |
| Fluency (1-5) | LLM-as-judge (Gemini rates output quality) |
| Instruction following | Exact format compliance rate |

---

## Project Structure

```
├── data/
│   ├── generate_instructions.py   # Synthetic instruction data via NVIDIA NIM
│   ├── split.py                    # Train/val splitter
│   ├── train.jsonl                 # Training set (instruction/output format)
│   └── val.jsonl                   # Validation set
├── training/
│   ├── train.py                    # Unsloth LoRA training script
│   └── utils.py                    # Formatting helpers
├── eval/
│   └── benchmark.py                # Instruction-following eval (BLEU + LLM-as-judge)
├── setup/
│   └── colab_setup.sh             # One-shot Colab environment setup
├── requirements.txt                # Full dependencies
├── requirements-colab.txt          # Colab-safe dependencies (GPU-free)
└── README.md
```

---

## Hardware

| Mode | Hardware | Time |
|------|----------|------|
| Training | Colab T4 (15 GB VRAM) | ~3-4 hrs |
| Inference (real-time) | GPU (A10/V100/T4) | ~1-2s per response |
| Inference (batch) | CPU via Ollama (Q4) | ~5-10 tok/s |

---

## Roadmap

- [x] Training pipeline (Unsloth + LoRA on T4)
- [x] Colab one-shot setup
- [x] CPU dry-run verification
- [ ] Instruction dataset generation via Nemotron-Super
- [ ] Training on 15K instruction records
- [ ] Evals: BLEU, ROUGE, LLM-as-judge
- [ ] HF model card + dataset release
- [ ] Gradio demo on HF Spaces

---

## License

MIT
