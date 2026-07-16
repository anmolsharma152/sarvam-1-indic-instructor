#!/usr/bin/env python3
"""
schema.py
Pydantic models for the Indic Instructor FastAPI serving layer.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class InferenceRequest(BaseModel):
    prompt: str = Field(
        ..., description="Raw instruction or full ChatML sequence"
    )
    max_tokens: int = Field(256, ge=1, le=2048, description="Max new tokens")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = Field(
        None, description="Optional stop strings"
    )


class InferenceResponse(BaseModel):
    text: str = Field(..., description="Generated assistant text")
    cached: bool = Field(
        False, description="True if served from the in-memory response cache"
    )


class StreamToken(BaseModel):
    text: str = Field(..., description="Token / chunk of generated text")
    index: int = Field(..., description="Chunk index in the stream")
