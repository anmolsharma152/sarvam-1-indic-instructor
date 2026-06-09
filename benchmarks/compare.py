#!/usr/bin/env python3
"""
compare.py
Runs comparison benchmarks between base and fine-tuned models on:
1. Accuracy (Role classification & Salary bucket prediction)
2. Generation latency & throughput (tokens/sec, TTFT)
Outputs comparison JSON files and generates visual performance charts.
"""
import os
import sys
import json
import argparse
import time
import torch
import matplotlib.pyplot as plt

# Add eval directory to path to import benchmark functions
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval import benchmark as eval_module

def measure_ttft(model, tokenizer, prompt: str, device: str) -> float:
    """Measure Time-to-First-Token (TTFT) for a single prompt."""
    formatted_prompt = eval_module.format_chatml_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Truncate if exceeds max_pos
    max_pos = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    max_input_len = max_pos - 5
    if inputs["input_ids"].shape[-1] > max_input_len:
        inputs["input_ids"] = inputs["input_ids"][:, -max_input_len:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_len:]
            
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warm up / run once to measure TTFT
    # TTFT is the latency to generate exactly 1 new token
    start_time = time.time()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    ttft = time.time() - start_time
    return ttft

def run_single_benchmark(model_id: str, adapter_path: str, test_file: str, num_samples: int, device: str, dry_run: bool):
    """Load model/adapter, run benchmark, and return metrics + raw predictions."""
    print("\n" + "="*50)
    print(f"BENCHMARKING MODEL: {model_id}")
    if adapter_path:
        print(f"PEFT ADAPTER: {adapter_path}")
    print("="*50)
    
    # Setup tokenizer and model
    tokenizer = eval_module.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if device == "cuda":
        model = eval_module.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        model = eval_module.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
    if adapter_path and not dry_run:
        from peft import PeftModel
        print(f"Applying adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    model.eval()
    
    # Load dataset
    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records = records[:min(num_samples, len(records))]
    
    y_true_role = []
    y_pred_role = []
    y_true_salary = []
    y_pred_salary = []
    
    latencies = []
    token_counts = []
    ttfts = []
    
    for idx, rec in enumerate(records):
        prompt_content = rec["prompt"]
        job_desc = prompt_content.split("Job Description:\n")[-1].replace("<|im_end|>\n<|im_start|>assistant\n", "").strip()
        
        completion_content = rec["completion"]
        true_role, true_sal = eval_module.parse_predictions(completion_content)
        y_true_role.append(true_role)
        y_true_salary.append(true_sal)
        
        prompt = eval_module.format_chatml_prompt(job_desc)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Truncate if exceeds max_pos
        max_pos = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
        max_input_len = max_pos - 65
        if inputs["input_ids"].shape[-1] > max_input_len:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_len:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_len:]
                
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Measure TTFT
        ttft = measure_ttft(model, tokenizer, job_desc, model.device)
        ttfts.append(ttft)
        
        # Measure Full Generation
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        latency = time.time() - start_time
        latencies.append(latency)
        
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_counts.append(len(generated_ids))
        
        pred_role, pred_sal = eval_module.parse_predictions(generated_text)
        
        if dry_run:
            pred_role = true_role if eval_module.random.random() > 0.3 else "Unknown"
            pred_sal = true_sal if eval_module.random.random() > 0.3 else "Unknown"
            
        y_pred_role.append(pred_role)
        y_pred_salary.append(pred_sal)
        
    role_acc = eval_module.accuracy_score(y_true_role, y_pred_role)
    salary_acc = eval_module.accuracy_score(y_true_salary, y_pred_salary)
    avg_latency = sum(latencies) / len(latencies)
    avg_ttft = sum(ttfts) / len(ttfts)
    tokens_per_sec = sum(token_counts) / sum(latencies) if sum(latencies) > 0 else 0
    
    return {
        "role_accuracy": role_acc,
        "salary_accuracy": salary_acc,
        "avg_ttft_seconds": avg_ttft,
        "avg_latency_seconds": avg_latency,
        "tokens_per_second": tokens_per_sec
    }

def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument("--model", type=str, default="sarvamai/sarvam-1", help="Base model ID")
    parser.add_argument("--adapter", type=str, default="models/sarvam-job-desc-lora", help="Path to adapter checkpoint")
    parser.add_argument("--test_file", type=str, default="data/val.jsonl", help="Validation dataset JSONL")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="benchmarks/results", help="Directory to save comparison files")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Dry run on CPU with tiny model")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples = args.num_samples
    model_id = args.model
    adapter_path = args.adapter
    
    if args.dry_run or device == "cpu":
        print("\n" + "="*80)
        print("WARNING: Running COMPARISON in DRY-RUN / CPU MODE.")
        print("Using tiny random model and 5 validation samples.")
        print("="*80 + "\n")
        model_id = "hf-internal-testing/tiny-random-gpt2"
        num_samples = 5
        # Ensure adapter path points to the dry-run output if it exists
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter path {adapter_path} not found. Running base model only comparison or using default folder.")
            
    # Run Base Model Benchmark
    base_metrics = run_single_benchmark(
        model_id=model_id,
        adapter_path=None,
        test_file=args.test_file,
        num_samples=num_samples,
        device=device,
        dry_run=args.dry_run
    )
    
    # Run Fine-tuned Model Benchmark
    ft_metrics = run_single_benchmark(
        model_id=model_id,
        adapter_path=adapter_path,
        test_file=args.test_file,
        num_samples=num_samples,
        device=device,
        dry_run=args.dry_run
    )
    
    # Format comparison output
    comparison = {
        "base_model": model_id,
        "adapter_model": adapter_path,
        "metrics_comparison": {
            "role_accuracy": {
                "base": base_metrics["role_accuracy"],
                "fine_tuned": ft_metrics["role_accuracy"],
                "gain": ft_metrics["role_accuracy"] - base_metrics["role_accuracy"]
            },
            "salary_accuracy": {
                "base": base_metrics["salary_accuracy"],
                "fine_tuned": ft_metrics["salary_accuracy"],
                "gain": ft_metrics["salary_accuracy"] - base_metrics["salary_accuracy"]
            },
            "avg_ttft_seconds": {
                "base": base_metrics["avg_ttft_seconds"],
                "fine_tuned": ft_metrics["avg_ttft_seconds"],
                "overhead_percent": ((ft_metrics["avg_ttft_seconds"] - base_metrics["avg_ttft_seconds"]) / base_metrics["avg_ttft_seconds"] * 100) if base_metrics["avg_ttft_seconds"] > 0 else 0
            },
            "tokens_per_second": {
                "base": base_metrics["tokens_per_second"],
                "fine_tuned": ft_metrics["tokens_per_second"],
                "change_percent": ((ft_metrics["tokens_per_second"] - base_metrics["tokens_per_second"]) / base_metrics["tokens_per_second"] * 100) if base_metrics["tokens_per_second"] > 0 else 0
            }
        }
    }
    
    print("\n" + "="*30 + " METRICS COMPARISON " + "="*30)
    print(f"Metric                 | Base Model  | Fine-Tuned  | Gain/Diff")
    print("-"*80)
    print(f"Role Accuracy          | {base_metrics['role_accuracy']:.4f}      | {ft_metrics['role_accuracy']:.4f}      | {comparison['metrics_comparison']['role_accuracy']['gain']:+.4f}")
    print(f"Salary Accuracy        | {base_metrics['salary_accuracy']:.4f}      | {ft_metrics['salary_accuracy']:.4f}      | {comparison['metrics_comparison']['salary_accuracy']['gain']:+.4f}")
    print(f"Avg TTFT (s)           | {base_metrics['avg_ttft_seconds']:.4f}s     | {ft_metrics['avg_ttft_seconds']:.4f}s     | {comparison['metrics_comparison']['avg_ttft_seconds']['overhead_percent']:+.1f}%")
    print(f"Throughput (tokens/s)  | {base_metrics['tokens_per_second']:.2f}        | {ft_metrics['tokens_per_second']:.2f}        | {comparison['metrics_comparison']['tokens_per_second']['change_percent']:+.1f}%")
    print("="*80 + "\n")
    
    # Write JSON report
    report_file = os.path.join(args.output_dir, "comparison_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    print(f"Comparison report saved to {report_file}")
    
    # Generate Comparison Charts
    metrics = ['Role Accuracy', 'Salary Accuracy']
    base_scores = [base_metrics['role_accuracy'], base_metrics['salary_accuracy']]
    ft_scores = [ft_metrics['role_accuracy'], ft_metrics['salary_accuracy']]
    
    x = range(len(metrics))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot accuracy on ax1
    rects1 = ax1.bar([i - width/2 for i in x], base_scores, width, label='Base Model', color='#a8dadc')
    rects2 = ax1.bar([i + width/2 for i in x], ft_scores, width, label='Fine-Tuned Model', color='#457b9d')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Performance Comparison: Base vs Fine-Tuned Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper left')
    
    # Add values on top of bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    
    plt.tight_layout()
    chart_file = os.path.join(args.output_dir, "comparison_chart.png")
    plt.savefig(chart_file, dpi=300)
    print(f"Comparison performance chart saved to {chart_file}")
    
    # Also save speed chart
    fig, ax2 = plt.subplots(figsize=(6, 4))
    speeds = [base_metrics['tokens_per_second'], ft_metrics['tokens_per_second']]
    models = ['Base Model', 'Fine-Tuned Model']
    
    bars = ax2.bar(models, speeds, color=['#e63946', '#1d3557'], width=0.5)
    ax2.set_ylabel('Tokens per Second')
    ax2.set_title('Generation Speed Comparison')
    ax2.set_ylim(0, max(speeds) * 1.2 if max(speeds) > 0 else 10)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.tight_layout()
    speed_chart_file = os.path.join(args.output_dir, "speed_comparison_chart.png")
    plt.savefig(speed_chart_file, dpi=300)
    print(f"Speed comparison chart saved to {speed_chart_file}")

if __name__ == "__main__":
    main()
