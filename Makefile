.PHONY: generate split train train-dry train-colab eval eval-dry compare compare-dry serve clean all

ADAPTER ?= models/sarvam-1-indic-instructor
MODEL ?= sarvamai/sarvam-1

# ─── Dataset ────────────────────────────────────────────────────────────────

generate:
	python data/generate_instructions.py --count 15000 --output data/raw_instructions.jsonl

split:
	python data/split.py --input data/raw_instructions.jsonl

# ─── Training ────────────────────────────────────────────────────────────────

train:
	python training/train.py --output_dir $(ADAPTER)

train-dry:
	python training/train.py --dry-run --output_dir $(ADAPTER)

train-colab:
	python training/train.py --model_id $(MODEL) \
		--train_file data/train.jsonl \
		--val_file data/val.jsonl \
		--output_dir $(ADAPTER) \
		--epochs 3 --batch_size 4

# ─── Evaluation ──────────────────────────────────────────────────────────────

eval:
	python eval/benchmark.py --model $(MODEL) --adapter $(ADAPTER) --test_file data/val.jsonl --num_samples 500

eval-dry:
	python eval/benchmark.py --dry-run --num_samples 5

compare:
	python benchmarks/compare.py --model $(MODEL) --adapter $(ADAPTER) --test_file data/val.jsonl --num_samples 50

compare-dry:
	python benchmarks/compare.py --dry-run --num_samples 5

# ─── Serving ─────────────────────────────────────────────────────────────────

serve:
	cd serving && python app.py --model $(MODEL) --adapter ../$(ADAPTER) --host 127.0.0.1 --port 8000

# ─── Utility ─────────────────────────────────────────────────────────────────

clean:
	rm -rf data/raw_instructions.jsonl data/train.jsonl data/val.jsonl
	rm -rf models/
	rm -rf eval/results.json
	rm -rf benchmarks/results/
	rm -rf logs/
	rm -rf wandb/
	rm -rf unsloth_compiled_cache/
	rm -rf __pycache__ */__pycache__ */*/__pycache__

# ─── Pipeline ────────────────────────────────────────────────────────────────

all: generate split
	@echo "Dataset ready. Run 'make train-colab' on Colab (or 'make train' with CUDA)."
