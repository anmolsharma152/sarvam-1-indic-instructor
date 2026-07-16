#!/usr/bin/env python3
"""
app.py
FastAPI inference for Indic Instructor.

Engines:
  1. vLLM when CUDA + vLLM are available (preferred for production).
  2. Hugging Face Transformers fallback (CPU / local verification).

Optional LoRA adapter (HF engine only) and an in-memory LRU response cache
for identical non-streaming /generate calls.
"""
import os
import hashlib
import json
import uuid
import argparse
import asyncio
from collections import OrderedDict
from threading import Thread, Lock
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from schema import InferenceRequest, InferenceResponse, StreamToken

try:
    import vllm
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

app = FastAPI(
    title="Indic Instructor API",
    description=(
        "Inference for Sarvam-1 instruction following "
        "(Hinglish / Hindi / English). Supports vLLM or HF + optional LoRA."
    ),
    version="1.0.0",
)

engine_type = None
hf_model = None
hf_tokenizer = None
vllm_engine = None
args = None

# Simple process-local LRU cache for /generate
_cache_lock = Lock()
_response_cache: OrderedDict = OrderedDict()
_cache_hits = 0
_cache_misses = 0


class ResponseCache:
    """Thread-safe LRU cache keyed by request fingerprint."""

    def __init__(self, maxsize: int = 256):
        self.maxsize = max(0, maxsize)

    def _key(self, req: InferenceRequest) -> str:
        payload = {
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stop_sequences": req.stop_sequences,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, req: InferenceRequest) -> str | None:
        global _cache_hits, _cache_misses
        if self.maxsize == 0:
            return None
        key = self._key(req)
        with _cache_lock:
            if key not in _response_cache:
                _cache_misses += 1
                return None
            _response_cache.move_to_end(key)
            _cache_hits += 1
            return _response_cache[key]

    def put(self, req: InferenceRequest, text: str) -> None:
        if self.maxsize == 0:
            return
        key = self._key(req)
        with _cache_lock:
            _response_cache[key] = text
            _response_cache.move_to_end(key)
            while len(_response_cache) > self.maxsize:
                _response_cache.popitem(last=False)

    def clear(self) -> int:
        with _cache_lock:
            n = len(_response_cache)
            _response_cache.clear()
            return n

    def stats(self) -> dict:
        with _cache_lock:
            return {
                "size": len(_response_cache),
                "maxsize": self.maxsize,
                "hits": _cache_hits,
                "misses": _cache_misses,
            }


response_cache = ResponseCache(maxsize=256)


def format_chatml_prompt(prompt: str) -> str:
    """Wrap raw instruction in ChatML if not already formatted."""
    if "<|im_start|>" in prompt:
        return prompt
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@app.on_event("startup")
async def startup_event():
    global engine_type, hf_model, hf_tokenizer, vllm_engine, args, response_cache

    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="sarvamai/sarvam-1")
        parser.add_argument("--adapter", type=str, default=None)
        parser.add_argument("--force_hf", action="store_true")
        parser.add_argument("--cache_size", type=int, default=256)
        args, _ = parser.parse_known_args()

    response_cache = ResponseCache(maxsize=getattr(args, "cache_size", 256))
    print(f"Loading weights: {args.model}")
    if getattr(args, "adapter", None):
        print(f"LoRA adapter: {args.adapter}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_adapter = bool(getattr(args, "adapter", None) and os.path.isdir(args.adapter))

    # vLLM does not load PEFT adapters here; require HF when adapter is set
    if VLLM_AVAILABLE and device == "cuda" and not args.force_hf and not use_adapter:
        print("Initializing vLLM engine...")
        engine_type = "vllm"
        engine_args = AsyncEngineArgs(
            model=args.model,
            max_model_len=2048,
            trust_remote_code=True,
        )
        vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    else:
        if use_adapter and VLLM_AVAILABLE and not args.force_hf:
            print("Adapter requested — using HF engine (vLLM needs a merged model path).")
        print("Initializing Hugging Face engine...")
        engine_type = "hf"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        if device == "cuda":
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        if use_adapter:
            from peft import PeftModel

            print(f"Applying PEFT adapter from {args.adapter}")
            hf_model = PeftModel.from_pretrained(hf_model, args.adapter)

        hf_model.eval()

    print(f"Startup complete. Engine: {engine_type.upper()} | cache_size={response_cache.maxsize}")


async def hf_stream_generator(prompt: str, req: InferenceRequest):
    from transformers import TextIteratorStreamer

    formatted_prompt = format_chatml_prompt(prompt)
    inputs = hf_tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(hf_tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=(req.temperature > 0.0),
        pad_token_id=hf_tokenizer.pad_token_id,
    )

    thread = Thread(target=hf_model.generate, kwargs=generation_kwargs)
    thread.start()

    token_idx = 0
    for new_text in streamer:
        token_obj = StreamToken(text=new_text, index=token_idx)
        yield f"data: {token_obj.json()}\n\n"
        token_idx += 1
        await asyncio.sleep(0.001)


async def vllm_stream_generator(prompt: str, req: InferenceRequest):
    formatted_prompt = format_chatml_prompt(prompt)

    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop_sequences,
    )
    request_id = str(uuid.uuid4())

    results_generator = vllm_engine.generate(
        formatted_prompt,
        sampling_params,
        request_id,
    )

    last_text_len = 0
    token_idx = 0
    async for request_output in results_generator:
        full_text = request_output.outputs[0].text
        new_text = full_text[last_text_len:]
        last_text_len = len(full_text)

        if new_text:
            token_obj = StreamToken(text=new_text, index=token_idx)
            yield f"data: {token_obj.json()}\n\n"
            token_idx += 1


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    """Blocking completion. Cached when prompt + sampling params match."""
    cached = response_cache.get(req)
    if cached is not None:
        return InferenceResponse(text=cached, cached=True)

    formatted_prompt = format_chatml_prompt(req.prompt)

    if engine_type == "hf":
        inputs = hf_tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=(req.temperature > 0.0),
                pad_token_id=hf_tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = hf_tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_cache.put(req, generated_text)
        return InferenceResponse(text=generated_text, cached=False)

    if engine_type == "vllm":
        sampling_params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop_sequences,
        )
        request_id = str(uuid.uuid4())

        results_generator = vllm_engine.generate(
            formatted_prompt,
            sampling_params,
            request_id,
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        generated_text = final_output.outputs[0].text
        response_cache.put(req, generated_text)
        return InferenceResponse(text=generated_text, cached=False)

    raise HTTPException(status_code=500, detail="Inference engine not initialized.")


@app.post("/stream")
async def stream(req: InferenceRequest):
    """SSE streaming. Not cached (partial tokens)."""
    if engine_type == "hf":
        return StreamingResponse(
            hf_stream_generator(req.prompt, req),
            media_type="text/event-stream",
        )
    if engine_type == "vllm":
        return StreamingResponse(
            vllm_stream_generator(req.prompt, req),
            media_type="text/event-stream",
        )
    raise HTTPException(status_code=500, detail="Inference engine not initialized.")


@app.get("/health")
async def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy",
        "engine": engine_type,
        "device": device,
        "vllm_available": VLLM_AVAILABLE,
        "model": getattr(args, "model", None) if args else None,
        "adapter": getattr(args, "adapter", None) if args else None,
        "cache": response_cache.stats(),
    }


@app.post("/cache/clear")
async def cache_clear():
    cleared = response_cache.clear()
    return {"cleared": cleared, "cache": response_cache.stats()}


def main():
    global args
    parser = argparse.ArgumentParser(description="Indic Instructor FastAPI server")
    parser.add_argument(
        "--model",
        type=str,
        default="sarvamai/sarvam-1",
        help="HF model ID or path to base / merged weights",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA adapter path (forces HF engine)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--force_hf", action="store_true")
    parser.add_argument(
        "--cache_size",
        type=int,
        default=256,
        help="LRU size for /generate responses (0 disables)",
    )

    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
