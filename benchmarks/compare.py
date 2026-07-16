#!/usr/bin/env python3
"""
compare.py
Compare base Sarvam-1 vs fine-tuned Indic Instructor adapter on:
1. Instruction quality (BLEU-1, ROUGE-L)
2. Generation latency & throughput (tokens/sec, TTFT)

Writes comparison_report.json and chart PNGs under benchmarks/results/.
"""
import os
import sys
import json
import argparse
import time
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval import benchmark as eval_module


def measure_ttft(model, tokenizer, instruction: str, system: str = "") -> float:
    """Time-to-first-token for a single instruction."""
    prompt = eval_module.format_prompt(instruction, system)
    inputs = tokenizer(prompt, return_tensors="pt")

    max_pos = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 1024
    )
    max_input_len = max_pos - 5
    if inputs["input_ids"].shape[-1] > max_input_len:
        inputs["input_ids"] = inputs["input_ids"][:, -max_input_len:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_len:]

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return time.time() - start_time


def load_records(test_file: str, num_samples: int, dry_run: bool) -> list:
    records = []
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    elif dry_run:
        records = [
            {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
            {"instruction": "Translate: 'Good morning' to Hindi", "output": "शुभ प्रभात"},
            {"instruction": "2+2 kya hai?", "output": "4."},
            {"instruction": "Capital of France?", "output": "Paris."},
            {"instruction": "Summarize: India is a large country.", "output": "India is large."},
        ]
    else:
        raise FileNotFoundError(f"Test file not found: {test_file}")
    return records[: min(num_samples, len(records))]


def run_single_benchmark(
    model_id: str,
    adapter_path: str | None,
    test_file: str,
    num_samples: int,
    dry_run: bool,
) -> dict:
    """Load model/adapter, run instruction-following benchmark, return metrics."""
    print("\n" + "=" * 50)
    print(f"BENCHMARKING MODEL: {model_id}")
    if adapter_path:
        print(f"PEFT ADAPTER: {adapter_path}")
    print("=" * 50)

    tokenizer = eval_module.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = eval_module.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        model = eval_module.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    if adapter_path and not dry_run and os.path.isdir(adapter_path):
        from peft import PeftModel

        print(f"Applying adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    records = load_records(test_file, num_samples, dry_run)

    bleu_scores = []
    rouge_scores = []
    latencies = []
    token_counts = []
    ttfts = []

    for rec in records:
        instruction = rec.get("instruction", "")
        reference = rec.get("output", "")
        system = rec.get("system", "")

        prompt = eval_module.format_prompt(instruction, system)
        inputs = tokenizer(prompt, return_tensors="pt")

        max_pos = getattr(model.config, "n_positions", None) or getattr(
            model.config, "max_position_embeddings", 1024
        )
        max_input_len = max_pos - 128
        if inputs["input_ids"].shape[-1] > max_input_len:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_len:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_len:]

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        ttfts.append(measure_ttft(model, tokenizer, instruction, system))

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latencies.append(time.time() - start_time)

        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_counts.append(len(generated_ids))

        if dry_run:
            generated_text = reference

        bleu_scores.append(eval_module.compute_bleu(reference, generated_text))
        rouge_scores.append(eval_module.compute_rouge_l(reference, generated_text))

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    tokens_per_sec = sum(token_counts) / sum(latencies) if sum(latencies) > 0 else 0.0

    return {
        "bleu_1_avg": avg_bleu,
        "rouge_l_f1_avg": avg_rouge,
        "avg_ttft_seconds": avg_ttft,
        "avg_latency_seconds": avg_latency,
        "tokens_per_second": tokens_per_sec,
    }


def _pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned Indic Instructor on instruction metrics"
    )
    parser.add_argument("--model", type=str, default="sarvamai/sarvam-1", help="Base model ID")
    parser.add_argument(
        "--adapter",
        type=str,
        default="models/sarvam-1-indic-instructor",
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument("--test_file", type=str, default="data/val.jsonl", help="Validation JSONL")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples")
    parser.add_argument(
        "--output_dir", type=str, default="benchmarks/results", help="Output directory"
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="CPU dry-run")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples = args.num_samples
    model_id = args.model
    adapter_path = args.adapter

    if args.dry_run or device == "cpu":
        print("\n" + "=" * 80)
        print("WARNING: Running COMPARISON in DRY-RUN / CPU MODE.")
        print("Using tiny random model and a small sample set.")
        print("=" * 80 + "\n")
        model_id = "hf-internal-testing/tiny-random-gpt2"
        num_samples = min(num_samples, 5)
        if not os.path.isdir(adapter_path):
            print(f"Note: adapter path {adapter_path} not found; FT run uses base weights only.")

    base_metrics = run_single_benchmark(
        model_id=model_id,
        adapter_path=None,
        test_file=args.test_file,
        num_samples=num_samples,
        dry_run=args.dry_run,
    )

    ft_metrics = run_single_benchmark(
        model_id=model_id,
        adapter_path=adapter_path,
        test_file=args.test_file,
        num_samples=num_samples,
        dry_run=args.dry_run,
    )

    comparison = {
        "base_model": model_id,
        "adapter_model": adapter_path,
        "metrics_comparison": {
            "bleu_1_avg": {
                "base": base_metrics["bleu_1_avg"],
                "fine_tuned": ft_metrics["bleu_1_avg"],
                "gain": ft_metrics["bleu_1_avg"] - base_metrics["bleu_1_avg"],
            },
            "rouge_l_f1_avg": {
                "base": base_metrics["rouge_l_f1_avg"],
                "fine_tuned": ft_metrics["rouge_l_f1_avg"],
                "gain": ft_metrics["rouge_l_f1_avg"] - base_metrics["rouge_l_f1_avg"],
            },
            "avg_ttft_seconds": {
                "base": base_metrics["avg_ttft_seconds"],
                "fine_tuned": ft_metrics["avg_ttft_seconds"],
                "overhead_percent": _pct_change(
                    ft_metrics["avg_ttft_seconds"], base_metrics["avg_ttft_seconds"]
                ),
            },
            "tokens_per_second": {
                "base": base_metrics["tokens_per_second"],
                "fine_tuned": ft_metrics["tokens_per_second"],
                "change_percent": _pct_change(
                    ft_metrics["tokens_per_second"], base_metrics["tokens_per_second"]
                ),
            },
        },
    }

    print("\n" + "=" * 30 + " METRICS COMPARISON " + "=" * 30)
    print(f"{'Metric':<24} | {'Base':>10} | {'Fine-tuned':>10} | {'Gain/Diff':>10}")
    print("-" * 72)
    print(
        f"{'BLEU-1':<24} | {base_metrics['bleu_1_avg']:10.4f} | "
        f"{ft_metrics['bleu_1_avg']:10.4f} | "
        f"{comparison['metrics_comparison']['bleu_1_avg']['gain']:+10.4f}"
    )
    print(
        f"{'ROUGE-L F1':<24} | {base_metrics['rouge_l_f1_avg']:10.4f} | "
        f"{ft_metrics['rouge_l_f1_avg']:10.4f} | "
        f"{comparison['metrics_comparison']['rouge_l_f1_avg']['gain']:+10.4f}"
    )
    print(
        f"{'Avg TTFT (s)':<24} | {base_metrics['avg_ttft_seconds']:10.4f} | "
        f"{ft_metrics['avg_ttft_seconds']:10.4f} | "
        f"{comparison['metrics_comparison']['avg_ttft_seconds']['overhead_percent']:+9.1f}%"
    )
    print(
        f"{'Throughput (tok/s)':<24} | {base_metrics['tokens_per_second']:10.2f} | "
        f"{ft_metrics['tokens_per_second']:10.2f} | "
        f"{comparison['metrics_comparison']['tokens_per_second']['change_percent']:+9.1f}%"
    )
    print("=" * 72 + "\n")

    report_file = os.path.join(args.output_dir, "comparison_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    print(f"Comparison report saved to {report_file}")

    # Quality chart
    metrics = ["BLEU-1", "ROUGE-L F1"]
    base_scores = [base_metrics["bleu_1_avg"], base_metrics["rouge_l_f1_avg"]]
    ft_scores = [ft_metrics["bleu_1_avg"], ft_metrics["rouge_l_f1_avg"]]
    x = range(len(metrics))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    rects1 = ax1.bar([i - width / 2 for i in x], base_scores, width, label="Base", color="#a8dadc")
    rects2 = ax1.bar(
        [i + width / 2 for i in x], ft_scores, width, label="Fine-tuned", color="#457b9d"
    )
    ax1.set_ylabel("Score")
    ax1.set_title("Indic Instructor: Base vs Fine-tuned Quality")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="upper left")

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    plt.tight_layout()
    chart_file = os.path.join(args.output_dir, "comparison_chart.png")
    plt.savefig(chart_file, dpi=300)
    print(f"Quality chart saved to {chart_file}")

    # Speed chart
    fig, ax2 = plt.subplots(figsize=(6, 4))
    speeds = [base_metrics["tokens_per_second"], ft_metrics["tokens_per_second"]]
    models = ["Base", "Fine-tuned"]
    bars = ax2.bar(models, speeds, color=["#e63946", "#1d3557"], width=0.5)
    ax2.set_ylabel("Tokens per Second")
    ax2.set_title("Generation Speed Comparison")
    ax2.set_ylim(0, max(speeds) * 1.2 if max(speeds) > 0 else 10)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    speed_chart_file = os.path.join(args.output_dir, "speed_comparison_chart.png")
    plt.savefig(speed_chart_file, dpi=300)
    print(f"Speed chart saved to {speed_chart_file}")


if __name__ == "__main__":
    main()
