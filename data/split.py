#!/usr/bin/env python3
"""
split.py
Splits raw instruction JSONL into train/val sets.
"""

import json
import argparse
import random


def main():
    parser = argparse.ArgumentParser(description="Split JSONL into train/val")
    parser.add_argument("--input", type=str, default="data/raw_instructions.jsonl", help="Input JSONL")
    parser.add_argument("--train", type=str, default="data/train.jsonl", help="Output train JSONL")
    parser.add_argument("--val", type=str, default="data/val.jsonl", help="Output val JSONL")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.train) or ".", exist_ok=True)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    random.shuffle(records)
    split = int(len(records) * (1 - args.val_ratio))
    train, val = records[:split], records[split:]
    
    with open(args.train, "w", encoding="utf-8") as f:
        for rec in train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    with open(args.val, "w", encoding="utf-8") as f:
        for rec in val:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"Split {len(records)} records: {len(train)} train / {len(val)} val")


if __name__ == "__main__":
    import os
    main()
