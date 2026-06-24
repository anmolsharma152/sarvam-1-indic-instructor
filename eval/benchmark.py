#!/usr/bin/env python3

import os
import re
import json
import argparse
import time
import random
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def format_prompt(instruction: str, system: str = "") -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{instruction}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def compute_bleu(reference: str, candidate: str) -> float:
    import collections
    import math
    
    ref_tokens = reference.strip().split()
    cand_tokens = candidate.strip().split()
    
    if len(cand_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # BLEU-1 (precision of unigrams)
    matches = sum(1 for t in cand_tokens if t in ref_tokens)
    precision = matches / len(cand_tokens)
    
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(cand_tokens))) if len(cand_tokens) > 0 else 0.0
    
    return bp * precision


def compute_rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L: longest common subsequence based."""
    ref_tokens = reference.strip().split()
    cand_tokens = candidate.strip().split()
    
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0
    
    # LCS length
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    
    precision = lcs / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
    recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on instruction-following tasks")
    parser.add_argument("--model", type=str, default="sarvamai/sarvam-1", help="Path or HF ID of base model")
    parser.add_argument("--adapter", type=str, default=None, help="Path to saved LoRA adapter checkpoint")
    parser.add_argument("--test_file", type=str, default="data/val.jsonl", help="Validation dataset JSONL")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="eval/results.json", help="Path to output results JSON")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Run on CPU with tiny model")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    model_id = args.model
    num_samples = args.num_samples
    
    if args.dry_run or device == "cpu":
        print("\n" + "="*80)
        print("WARNING: Running in DRY-RUN / CPU MODE.")
        print("="*80 + "\n")
        model_id = "hf-internal-testing/tiny-random-gpt2"
        num_samples = min(num_samples, 5)
    
    if device == "cuda" and not args.dry_run:
        from unsloth import FastLanguageModel
        print(f"Loading model with Unsloth: {model_id}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        if args.adapter:
            print(f"Applying adapter: {args.adapter}...")
            model = PeftModel.from_pretrained(model, args.adapter)
        model = FastLanguageModel.for_inference(model)
    else:
        print(f"Loading fallback tokenizer and model: {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=None, torch_dtype=torch.float32, trust_remote_code=True
            )
        if args.adapter and not args.dry_run:
            print(f"Applying adapter: {args.adapter}...")
            model = PeftModel.from_pretrained(model, args.adapter)
    
    model.eval()
    
    # Load dataset
    print(f"Loading evaluation dataset: {args.test_file}...")
    records = []
    if os.path.exists(args.test_file):
        with open(args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        if args.dry_run:
            records = [
                {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
                {"instruction": "Translate: 'Good morning' to Hindi", "output": "शुभ प्रभात"},
                {"instruction": "2+2 kya hai?", "output": "4."},
                {"instruction": "Capital of France?", "output": "Paris."},
                {"instruction": "Summarize: India is a large country.", "output": "India is large."},
            ]
        else:
            print(f"Error: {args.test_file} not found.")
            return
    
    records = records[:min(num_samples, len(records))]
    print(f"Evaluating on {len(records)} samples...")
    
    y_true = []
    y_pred = []
    bleu_scores = []
    rouge_scores = []
    latencies = []
    token_counts = []
    
    for idx, rec in enumerate(tqdm(records)):
        instruction = rec.get("instruction", "")
        reference = rec.get("output", "")
        system = rec.get("system", "")
        
        y_true.append(reference)
        
        prompt = format_prompt(instruction, system)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        max_pos = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
        max_input_len = max_pos - 128
        if inputs["input_ids"].shape[-1] > max_input_len:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_len:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_len:]
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = time.time() - start_time
        latencies.append(latency)
        
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_counts.append(len(generated_ids))
        
        # Dry run mock
        if args.dry_run:
            generated_text = reference
        
        y_pred.append(generated_text)
        
        bleu = compute_bleu(reference, generated_text)
        rouge = compute_rouge_l(reference, generated_text)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    tokens_per_sec = sum(token_counts) / sum(latencies) if sum(latencies) > 0 else 0.0
    
    print("\n" + "="*40 + " RESULTS " + "="*40)
    print(f"BLEU-1 (avg):              {avg_bleu:.4f}")
    print(f"ROUGE-L F1 (avg):          {avg_rouge:.4f}")
    print(f"Average Latency:           {avg_latency:.4f}s")
    print(f"Average Generated Tokens:  {avg_tokens:.1f}")
    print(f"Throughput:                {tokens_per_sec:.2f} tok/s")
    print("="*89 + "\n")
    
    results = {
        "model": model_id,
        "adapter": args.adapter,
        "metrics": {
            "bleu_1_avg": avg_bleu,
            "rouge_l_f1_avg": avg_rouge,
            "avg_latency_seconds": avg_latency,
            "avg_generated_tokens": avg_tokens,
            "tokens_per_second": tokens_per_sec,
        },
        "samples": [
            {"instruction": r["instruction"], "reference": r["output"], "prediction": y_pred[i]}
            for i, r in enumerate(records[:10])
        ],
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
