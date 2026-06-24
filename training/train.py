#!/usr/bin/env python3

import os
import sys
import argparse
import torch

try:
    from unsloth import FastLanguageModel
    _UNSLOTH_AVAILABLE = True
except ImportError:
    _UNSLOTH_AVAILABLE = False

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def format_instruction(example: dict) -> str:
    system = example.get("system", "")
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{instruction}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{output}<|im_end|>")
    
    return "\n".join(parts)


def format_prompt_only(instruction: str, system: str = "") -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{instruction}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sarvam-1 for instruction following")
    parser.add_argument("--model_id", type=str, default="sarvamai/sarvam-1", help="HF model ID")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl", help="Path to training jsonl")
    parser.add_argument("--val_file", type=str, default="data/val.jsonl", help="Path to validation jsonl")
    parser.add_argument("--output_dir", type=str, default="models/sarvam-1-indic-instructor", help="Output dir")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Fast CPU test with tiny model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--wandb_project", type=str, default="sarvam-1-indic-instructor", help="W&B project")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    model_id = args.model_id
    epochs = args.epochs
    max_steps = -1
    use_unsloth = (device == "cuda" and not args.dry_run)
    
    if args.dry_run or device == "cpu":
        print("\n" + "="*80)
        print("WARNING: Running in DRY-RUN / CPU MODE.")
        print("Using tiny random model and minimal data steps.")
        print("="*80 + "\n")
        model_id = "hf-internal-testing/tiny-random-gpt2"
        epochs = 1
        max_steps = 3
        args.batch_size = 2
        os.environ["WANDB_MODE"] = "offline"
        print("W&B set to offline mode.")
    else:
        if not os.environ.get("WANDB_API_KEY"):
            print("WANDB_API_KEY not found. Defaulting W&B to offline mode.")
            os.environ["WANDB_MODE"] = "offline"
    
    # Load dataset
    print(f"Loading dataset from {args.train_file} and {args.val_file}...")
    dataset_files = {"train": args.train_file, "validation": args.val_file}
    try:
        dataset = load_dataset("json", data_files=dataset_files)
    except FileNotFoundError:
        if args.dry_run:
            print("Dataset not found. Creating synthetic dry-run data...")
            fake_train = Dataset.from_list([
                {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
                {"instruction": "2+2 kya hai?", "output": "4."},
                {"instruction": "Translate: 'Good morning' to Hindi", "output": "शुभ प्रभात"},
                {"instruction": "Machine learning kya hai?", "output": "Machine learning ek technique hai jahan computer data se seekhta hai."},
                {"instruction": "Summatize: India is a large country.", "output": "India is large."},
                {"instruction": "Cricket mein kitne player hote hain?", "output": "11 players."},
                {"instruction": "What is Python?", "output": "Python ek programming language hai."},
                {"instruction": "Explain gravity.", "output": "Gravity ek force hai jo cheezon ko neeche kheenchti hai."},
                {"instruction": "Capital of France?", "output": "Paris."},
                {"instruction": "Write a poem on nature.", "output": "Nature is beautiful, har taraf hara bhara, pankhion ka geet, bahar ka nazara."},
            ])
            fake_val = Dataset.from_list(fake_train[:5])
            dataset = {"train": fake_train, "validation": fake_val}
        else:
            raise
    
    if args.dry_run or device == "cpu":
        if "train" in dataset and len(dataset["train"]) > 10:
            dataset["train"] = dataset["train"].select(range(min(10, len(dataset["train"]))))
            dataset["validation"] = dataset["validation"].select(range(min(5, len(dataset["validation"]))))
    
    print(f"Dataset loaded. Train size: {len(dataset['train'])}, Val size: {len(dataset['validation'])}")
    
    # Model loading
    if use_unsloth:
        print(f"Initializing Unsloth FastLanguageModel for: {model_id}...")
        if not _UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth is not installed. Install it: pip install unsloth")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        
        print("Injecting optimized Unsloth LoRA parameters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        peft_config = None
    else:
        print(f"Loading standard fallback tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        print(f"Loading standard fallback model: {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=None,
            device_map=None,
            trust_remote_code=True,
        )
        
        target_modules = ["c_attn", "c_proj"] if "gpt2" in model_id.lower() else ["q_proj", "v_proj", "k_proj", "o_proj"]
        print(f"Configuring standard LoRA PEFT targeting: {target_modules}")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100 if max_steps == -1 else 1,
        logging_steps=10 if max_steps == -1 else 1,
        save_strategy="steps",
        save_steps=200 if max_steps == -1 else 2,
        save_total_limit=2,
        load_best_model_at_end=True if max_steps == -1 else False,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        use_cpu=(device == "cpu"),
        report_to="wandb",
        run_name=f"sarvam-1-indic-instructor-{device}" if not args.dry_run else "indic-instructor-dry-run",
        logging_dir="./logs",
    )
    
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer if use_unsloth else None,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        args=training_args,
        max_seq_length=512 if args.dry_run else 2048,
        formatting_func=format_instruction,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final adapter model weights to {args.output_dir}...")
    if use_unsloth:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        merged_dir = args.output_dir + "-merged-16bit"
        print(f"Saving merged 16-bit model to {merged_dir}...")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    else:
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
