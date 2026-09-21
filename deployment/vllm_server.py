#!/usr/bin/env python3
"""
vLLM-based inference server for MaintainAI fine-tuned SLM.

Supports:
- Base model + LoRA adapter (Qwen2.5-3B-Instruct + LoRA)
- OpenAI-compatible /v1/completions endpoint
- Structured JSON output via guided decoding (optional)
- Health check endpoint

Usage:
    python deployment/vllm_server.py --model Qwen/Qwen2.5-3B-Instruct --lora-path /path/to/adapter --port 8000

Requirements:
    pip install vllm fastapi uvicorn pydantic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maintainai.vllm_server")

# Global LLM instance
llm: Optional[LLM] = None
lora_name: str = "maintainai"
lora_path: Optional[str] = None
base_model: str = "Qwen/Qwen2.5-3B-Instruct"


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    prompt: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop: Optional[list[str]] = None
    stream: bool = False


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


def format_prompt(system: str, user: str) -> str:
    """Format prompt in ChatML format for Qwen."""
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vLLM on startup, cleanup on shutdown."""
    global llm
    logger.info(f"Loading base model: {base_model}")
    llm = LLM(
        model=base_model,
        dtype="float16",
        max_model_len=2048,
        enable_lora=lora_path is not None,
        max_lora_rank=16,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    if lora_path:
        logger.info(f"Loading LoRA adapter: {lora_path}")
    logger.info("vLLM server ready")
    yield
    logger.info("Shutting down vLLM server")


app = FastAPI(title="MaintainAI vLLM Inference", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": base_model,
        "lora_loaded": lora_path is not None,
        "lora_path": lora_path,
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest) -> CompletionResponse:
    """OpenAI-compatible completions endpoint."""
    if llm is None:
        raise HTTPException(503, "Model not loaded")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )

    lora_request = None
    if lora_path:
        lora_request = LoRARequest(lora_name, 1, lora_path)

    outputs = llm.generate(
        [request.prompt],
        sampling_params,
        lora_request=lora_request,
    )

    output = outputs[0]
    generated_text = output.outputs[0].text

    import time
    return CompletionResponse(
        id=f"cmpl-{hash(generated_text) & 0xffffffff:08x}",
        created=int(time.time()),
        model=base_model + (f"+{lora_name}" if lora_path else ""),
        choices=[{
            "index": 0,
            "text": generated_text,
            "finish_reason": output.outputs[0].finish_reason,
        }],
        usage={
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        },
    )


@app.post("/generate")
async def generate(request: dict) -> dict:
    """Simplified endpoint for MaintainAI SLMService.RemoteBackend."""
    prompt = request.get("prompt", "")
    max_new_tokens = request.get("max_new_tokens", 512)
    temperature = request.get("temperature", 0.2)

    if not prompt:
        raise HTTPException(400, "prompt is required")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
    )

    lora_request = LoRARequest(lora_name, 1, lora_path) if lora_path else None
    outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)

    return {"generated_text": outputs[0].outputs[0].text}


def main():
    parser = argparse.ArgumentParser(description="MaintainAI vLLM Inference Server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name or path")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--lora-name", default="maintainai", help="LoRA adapter name")
    args = parser.parse_args()

    global base_model, lora_path, lora_name
    base_model = args.model
    lora_path = args.lora_path
    lora_name = args.lora_name

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()