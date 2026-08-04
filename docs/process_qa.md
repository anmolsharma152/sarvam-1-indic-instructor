# Project Process & Architecture Q&A

> Summary of key questions and technical clarifications regarding the synthetic dataset generation pipeline, execution process, output structure, and model lineage.

---

### Q1: Will the generation script automatically stop when it reaches 15,000 entries?
**Answer:**  
Yes, absolutely. `data/generate_instructions.py` enforces target termination in two ways:
1. At startup, it checks existing lines in `--output` and only generates seeds for `args.count - existing_count`.
2. In the batch loop, it tracks `total = existing_count + len(all_records)` and executes `if total >= args.count: break`.
3. Once 15,000 records are written, `main()` prints the language breakdown summary and exits with code `0`.

---

### Q2: What file is this script writing the data to?
**Answer:**  
The data is appended to `data/raw_instructions.jsonl` (JSON Lines format). Each line is an independent, valid JSON object written in append mode (`"a"`) with per-record flushing (`f.flush()`) to ensure no data loss even during process interruptions.

---

### Q3: Does each entry have both an `instruction` and an `output`?
**Answer:**  
Yes. 100% of saved records strictly contain both `"instruction"` and `"output"` keys. The script explicitly validates this condition before writing:
```python
if record and "instruction" in record and "output" in record:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```
Any malformed response or missing field causes the record to be discarded, guaranteeing zero missing/null fields in the dataset.

---

### Q4: What is the best use case for this dataset?
**Answer:**  
Because the dataset consists of single-turn instruction-output pairs without function schemas or execution traces, it is **not for tool calling or complex agentic multi-step reasoning**. 

Its primary best use cases are:
1. **Hinglish & Multilingual Chatbots:** Teaching small Indic models (like Sarvam-1 2B) how to follow instructions in Roman Hinglish (`"Mera order update do..."`), Hindi, and English.
2. **Code-Switched Summarization & Q&A:** Cross-lingual tasks like summarizing English articles into Hinglish or answering Roman Hinglish queries using Hindi context.
3. **Indian Content Generation & Rewriting:** Normalizing informal Hinglish into clean Hindi/English or generating social media copy.

---

### Q5: What model is generating the dataset? Is it Llama or Nemotron?
**Answer:**  
The generator model ID is **`nvidia/llama-3.3-nemotron-super-49b-v1`**.

It is **both Llama and Nemotron**:
* **Llama 3.3 (Meta):** The base foundational model architecture and tokenizer.
* **Nemotron (NVIDIA):** NVIDIA's custom 49B parameter pruning, distillation, and SteerLM alignment.

Think of Llama 3.3 as the base engine and NVIDIA Nemotron as NVIDIA tuning that engine into a specialized high-performance model.

---

### Q6: Why call the dataset `sarvam-1-indic-instructor-15k` vs `nemotron-49b-indic-instructions-15k`?
**Answer:**  
- **`sarvam-1-indic-instructor-15k`** names the dataset after your **downstream target project** (fine-tuning Sarvam-1 2B).
- **`nemotron-49b-indic-instructions-15k`** names the dataset after the **teacher/generator model** (NVIDIA Nemotron 49B).

Both naming schemes are valid. When publishing to Hugging Face Datasets, highlighting Nemotron 49B in the dataset name or description makes the high quality of the teacher model immediately obvious to the community.
