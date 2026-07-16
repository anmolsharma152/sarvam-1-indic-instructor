# Unsloth training notes (Indic Instructor)

How this repo trains Sarvam-1 with Unsloth vs a plain HF TRL path.

## LoRA targets

Unsloth path targets attention **and** MLP modules:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

CPU / dry-run fallback (tiny GPT-2) uses architecture-specific modules such as `c_attn` / `c_proj`.

## VRAM / speed knobs

| Setting | Value | Why |
|---------|-------|-----|
| `load_in_4bit` | `True` | Native Unsloth 4-bit load |
| `lora_dropout` | `0` under Unsloth | Faster; less mask overhead |
| Gradient checkpointing | `"unsloth"` | Lower activation memory |
| Precision | `bf16` if supported else `fp16` | Hardware-aware |

## Saving

- Adapter: `model.save_pretrained(output_dir)`
- Optional merge: `model.save_pretrained_merged(..., save_method="merged_16bit")` for vLLM-friendly weights

## Chat format

Training examples are formatted as ChatML:

```text
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

Optional `system` field is supported when present in the JSONL.
