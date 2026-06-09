#!/usr/bin/env python3
"""
app.py
FastAPI server for streaming inference. Supports dual-engines:
1. vLLM Engine (when CUDA/vLLM is available, optimal for production GPU).
2. Hugging Face Transformers Engine (fallback for CPU and local verification).
"""
import os
import json
import uuid
import argparse
import asyncio
from threading import Thread
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import schemas
from schema import InferenceRequest, InferenceResponse, StreamToken

# Detect vLLM availability
try:
    import vllm
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

app = FastAPI(
    title="Sarvam-1 Job Description Inference API",
    description="FastAPI endpoint for streaming role classification and salary bucket predictions.",
    version="1.0.0"
)

# Global model/tokenizer/engine variables
engine_type = None
hf_model = None
hf_tokenizer = None
vllm_engine = None

# CLI arguments parsed during startup
args = None

def format_chatml_prompt(prompt: str) -> str:
    """Format raw job description into the expected ChatML format if needed."""
    if "<|im_start|>" in prompt:
        return prompt
        
    system_prompt = "You are a professional HR assistant specializing in parsing and classifying Indian job postings."
    user_prompt = (
        f"Analyze the following job description. Classify it into a Role Category "
        f"(Software Engineering, Data Science, Product Management, Marketing, HR, Finance) "
        f"and determine its Salary Bucket (Entry, Mid, Senior, Executive).\n\n"
        f"Job Description:\n{prompt}"
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

@app.on_event("startup")
async def startup_event():
    global engine_type, hf_model, hf_tokenizer, vllm_engine, args
    
    # If starting via uvicorn directly, args won't be initialized by CLI. Use defaults.
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="sarvamai/sarvam-1")
        parser.add_argument("--adapter", type=str, default=None)
        parser.add_argument("--force_hf", action="store_true")
        args, _ = parser.parse_known_args()
        
    print(f"Loading weights with model path/id: {args.model}")
    if args.adapter:
        print(f"Loading PEFT adapter checkpoint: {args.adapter}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Decide which engine to initialize
    if VLLM_AVAILABLE and device == "cuda" and not args.force_hf:
        print("Initializing production vLLM Engine...")
        engine_type = "vllm"
        engine_args = AsyncEngineArgs(
            model=args.model,
            enable_lora=(args.adapter is not None),
            max_loras=1 if args.adapter else 0,
            max_model_len=1024,
            trust_remote_code=True
        )
        vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    else:
        print("Initializing standard Hugging Face Transformers Engine (fallback)...")
        engine_type = "hf"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
            
        print("Loading HF Model...")
        if device == "cuda":
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
        if args.adapter:
            from peft import PeftModel
            print("Applying PEFT/LoRA adapter weights to HF model...")
            hf_model = PeftModel.from_pretrained(hf_model, args.adapter)
            
        # Ensure model is in evaluation mode
        hf_model.eval()
        
    print(f"Startup complete. Selected engine: {engine_type.upper()}")

async def hf_stream_generator(prompt: str, req: InferenceRequest):
    """Generates tokens sequentially using Hugging Face TextIteratorStreamer."""
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
        pad_token_id=hf_tokenizer.pad_token_id
    )
    
    # Run in a background thread to prevent blocking the async loop
    thread = Thread(target=hf_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    token_idx = 0
    for new_text in streamer:
        # SSE format: data: {...}\n\n
        token_obj = StreamToken(text=new_text, index=token_idx)
        yield f"data: {token_obj.json()}\n\n"
        token_idx += 1
        # Brief yield to yield execution to loop
        await asyncio.sleep(0.001)

async def vllm_stream_generator(prompt: str, req: InferenceRequest):
    """Generates tokens sequentially using vLLM's AsyncLLMEngine."""
    formatted_prompt = format_chatml_prompt(prompt)
    
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop_sequences
    )
    request_id = str(uuid.uuid4())
    
    lora_request = None
    if args.adapter:
        # ID 1, mapping to adapter folder
        lora_request = LoRARequest("job_desc_adapter", 1, args.adapter)
        
    results_generator = vllm_engine.generate(
        formatted_prompt, 
        sampling_params, 
        request_id, 
        lora_request=lora_request
    )
    
    last_text_len = 0
    token_idx = 0
    async for request_output in results_generator:
        # vLLM outputs are cumulative, calculate the delta chunk
        full_text = request_output.outputs[0].text
        new_text = full_text[last_text_len:]
        last_text_len = len(full_text)
        
        if new_text:
            token_obj = StreamToken(text=new_text, index=token_idx)
            yield f"data: {token_obj.json()}\n\n"
            token_idx += 1

@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    """Standard blocking endpoint to generate output in one go."""
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
                pad_token_id=hf_tokenizer.pad_token_id
            )
            
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = hf_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return InferenceResponse(text=generated_text)
        
    elif engine_type == "vllm":
        sampling_params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop_sequences
        )
        request_id = str(uuid.uuid4())
        
        lora_request = None
        if args.adapter:
            lora_request = LoRARequest("job_desc_adapter", 1, args.adapter)
            
        results_generator = vllm_engine.generate(
            formatted_prompt, 
            sampling_params, 
            request_id, 
            lora_request=lora_request
        )
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        generated_text = final_output.outputs[0].text
        return InferenceResponse(text=generated_text)
        
    else:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

@app.post("/stream")
async def stream(req: InferenceRequest):
    """SSE endpoint for streaming token completions."""
    if engine_type == "hf":
        return StreamingResponse(
            hf_stream_generator(req.prompt, req),
            media_type="text/event-stream"
        )
    elif engine_type == "vllm":
        return StreamingResponse(
            vllm_stream_generator(req.prompt, req),
            media_type="text/event-stream"
        )
    else:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

@app.get("/health")
async def health():
    """Health check endpoint returning system status."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy",
        "engine": engine_type,
        "device": device,
        "vllm_available": VLLM_AVAILABLE
    }

def main():
    global args
    parser = argparse.ArgumentParser(description="Start FastAPI serving engine")
    parser.add_argument("--model", type=str, default="sarvamai/sarvam-1", help="Path or Hugging Face ID of the base model")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter weights (optional)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--force_hf", action="store_true", help="Force HF Transformers engine even if vLLM is available")
    
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
