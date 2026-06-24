#!/usr/bin/env python3
"""
generate_instructions.py
Generates synthetic instruction-following data in Hinglish, Hindi, and English
using NVIDIA Nemotron-Super via the NIM API (OpenAI-compatible endpoint).

Usage:
    export NVIDIA_API_KEY="nvapi-..."
    python data/generate_instructions.py --count 15000 --output data/raw_instructions.jsonl

Output format:
    {"instruction": "...", "output": "..."}
    {"system": "...", "instruction": "...", "output": "..."}
"""

import os
import json
import argparse
import time
from openai import OpenAI

# ─── Seed instructions for each language ─────────────────────────────────────

SEED_TASKS = {
    "translation": {
        "prompt": "Generate a {lang} instruction asking to translate text {from_lang} to {to_lang}, then provide the correct translation.",
        "count": 0
    },
    "summarization": {
        "prompt": "Generate a {lang} instruction asking to summarize a paragraph, then provide a summary.",
        "count": 0
    },
    "qa": {
        "prompt": "Generate a {lang} question-answer pair about {topic}.",
        "count": 0
    },
    "brainstorming": {
        "prompt": "Generate a {lang} instruction asking for ideas about {topic}, then provide 3-5 ideas.",
        "count": 0
    },
    "classification": {
        "prompt": "Generate a {lang} instruction asking to classify something, then provide the classification.",
        "count": 0
    },
    "creative": {
        "prompt": "Generate a {lang} creative writing instruction, then provide a short story or poem.",
        "count": 0
    },
    "grammar": {
        "prompt": "Generate a {lang} instruction asking to correct grammar in a sentence, then provide the corrected version.",
        "count": 0
    },
}


def build_client():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY not set. Get one at https://build.nvidia.com/nim"
        )
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def generate_batch(client, model: str, seed_prompts: list[str], temperature: float = 0.8) -> list[dict]:
    results = []
    for seed in seed_prompts:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a dataset generator. Generate one instruction-output pair "
                            "in the specified language and script. "
                            "Output ONLY valid JSON with keys 'instruction' and 'output'. "
                            "Use Hinglish (Hindi+English mix in Roman script), Hindi (Devanagari), "
                            "or English as requested."
                        ),
                    },
                    {"role": "user", "content": seed},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            text = resp.choices[0].message.content.strip()
            # Try parsing JSON from response
            record = try_parse_json(text)
            if record and "instruction" in record and "output" in record:
                results.append(record)
            else:
                print(f"  [warn] Failed to parse: {text[:80]}...")
        except Exception as e:
            print(f"  [error] {e}")
            time.sleep(2)
    return results


def try_parse_json(text: str) -> dict | None:
    import re
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try extracting first JSON object
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def build_seed_prompts(count: int, languages: list[str], scripts: list[str]) -> list[str]:
    """Build a diverse set of seed prompts for generation."""
    topics = [
        "machine learning", "artificial intelligence", "cloud computing",
        "climate change", "Indian economy", "cricket", "Bollywood",
        "Indian festivals", "technology", "education",
        "healthcare in India", "startups", "remote work", "data science",
    ]
    combos = []
    for lang in languages:
        for script in scripts:
            if lang == "hi" and script == "roman":
                continue  # Hindi in Roman script is Hinglish or transliterated
            combos.append((lang, script))

    prompts = []
    task_types = list(SEED_TASKS.keys())
    for i in range(count):
        lang, script = combos[i % len(combos)]
        task = task_types[i % len(task_types)]
        topic = topics[i % len(topics)]
        
        lang_label = {"hi": "Hindi (Devanagari)", "en": "English", "hinglish": "Hinglish (Roman)"}.get(lang, lang)
        
        prompt = (
            f"Generate a {lang_label} instruction-output pair. "
            f"Task type: {task}. "
            f"Topic: {topic}. "
            f"Make the instruction realistic and natural. "
            f"Output as JSON with keys 'instruction' and 'output'."
        )
        prompts.append(prompt)
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate instruction data via NVIDIA NIM")
    parser.add_argument("--count", type=int, default=1000, help="Number of records to generate")
    parser.add_argument("--output", type=str, default="data/raw_instructions.jsonl", help="Output JSONL path")
    parser.add_argument("--languages", nargs="+", default=["hinglish", "hi", "en"],
                        choices=["hinglish", "hi", "en"], help="Languages to generate")
    parser.add_argument("--scripts", nargs="+", default=["roman", "devanagari"],
                        choices=["roman", "devanagari"], help="Scripts to use")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-super-49b-v1",
                        help="NVIDIA NIM model ID")
    parser.add_argument("--batch_size", type=int, default=10, help="Records per API call")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Connecting to NVIDIA NIM...")
    client = build_client()
    
    print(f"Building {args.count} seed prompts...")
    seed_prompts = build_seed_prompts(args.count, args.languages, args.scripts)
    
    print(f"Generating {args.count} records (batch_size={args.batch_size})...")
    all_records = []
    for i in range(0, len(seed_prompts), args.batch_size):
        batch = seed_prompts[i : i + args.batch_size]
        records = generate_batch(client, args.model, batch, args.temperature)
        all_records.extend(records)
        print(f"  Progress: {len(all_records)}/{args.count} records")
        
        if len(all_records) >= args.count:
            break
    
    # Trim to exact count
    all_records = all_records[:args.count]
    
    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\nDone! {len(all_records)} records saved to {args.output}")
    lang_counts = {}
    for rec in all_records:
        inst = rec.get("instruction", "")
        if any("\u0900" <= c <= "\u097F" for c in inst):
            tag = "hindi"
        elif any(c in inst for c in "ko hai se mein ka ki"):
            tag = "hinglish"
        else:
            tag = "english"
        lang_counts[tag] = lang_counts.get(tag, 0) + 1
    print(f"Language breakdown: {lang_counts}")


if __name__ == "__main__":
    main()
