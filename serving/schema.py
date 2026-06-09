#!/usr/bin/env python3
"""
schema.py
Pydantic data schemas for validation in the FastAPI serving layer.
"""
from pydantic import BaseModel, Field
from typing import Optional, List

class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="The input text or formatted ChatML sequence for generation")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability threshold")
    stop_sequences: Optional[List[str]] = Field(None, description="Optional list of stop tokens")

class InferenceResponse(BaseModel):
    text: str = Field(..., description="Generated text response from the model")

class StreamToken(BaseModel):
    text: str = Field(..., description="A chunk of generated text token")
    index: int = Field(..., description="Index sequence of the generated token")
