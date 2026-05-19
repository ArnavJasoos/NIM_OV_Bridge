#!/usr/bin/env python3
"""
OpenVINO GenAI OpenAI-Compatible API Server
Replaces NIM's Triton-backed FastAPI layer for Intel iGPU inference.
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Optional, List, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

try:
    import openvino_genai as ov_genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("[WARN] openvino_genai not found.")

try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    HAS_OPTIMUM = True
except ImportError:
    HAS_OPTIMUM = False

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/models"))
DEVICE = os.environ.get("OPENVINO_DEVICE", "GPU")
API_PORT = int(os.environ.get("API_PORT", "8000"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MODEL_NAME = os.environ.get("NIM_MODEL_NAME", "openvino-model")

app = FastAPI(title="NIM-OV-Bridge API Server", version="3.0.0")

pipeline = None
tokenizer = None
model_loaded = False


def load_model():
    global pipeline, tokenizer, model_loaded
    if not MODEL_PATH.exists():
        print(f"[FATAL] Model path not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"[LOAD] Loading model from {MODEL_PATH} on device: {DEVICE}")
    start = time.time()

    try:
        if HAS_GENAI:
            pipeline = ov_genai.LLMPipeline(str(MODEL_PATH), DEVICE)
            tokenizer = None
            print("[LOAD] OpenVINO GenAI pipeline ready.")
        elif HAS_OPTIMUM:
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
            pipeline = OVModelForCausalLM.from_pretrained(
                str(MODEL_PATH), device=DEVICE, trust_remote_code=True
            )
            print("[LOAD] Optimum-Intel model ready.")
        else:
            raise RuntimeError("No OpenVINO backend available.")
        model_loaded = True
        print(f"[LOAD] Completed in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"[FATAL] Model load failed: {e}")
        raise


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=MAX_MODEL_LEN)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=MAX_MODEL_LEN)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nim-ov-bridge"


def format_chat_prompt(messages: List[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"|<|system|>\n{msg.content}")
        elif msg.role == "user":
            parts.append(f"|<|user|>\n{msg.content}")
        elif msg.role == "assistant":
            parts.append(f"|<|assistant|>\n{msg.content}")
    parts.append("|<|assistant|>\n")
    return "\n".join(parts)


def make_chunk(text: str, model: str, finish_reason: Optional[str] = None, index: int = 0) -> str:
    chunk = {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": index,
            "delta": {"content": text} if text else {},
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"


def make_final_chunk(model: str, finish_reason: str = "stop") -> str:
    return make_chunk("", model, finish_reason) + "data: [DONE]\n\n"


def make_completion_response(text: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
@app.get("/health/ready")
@app.get("/health/live")
async def health():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": DEVICE, "model": MODEL_NAME}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [ModelInfo(id=MODEL_NAME).model_dump()]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prompt = format_chat_prompt(request.messages)
    if request.stream:
        return StreamingResponse(chat_stream_generator(prompt, request), media_type="text/event-stream")
    else:
        return await chat_non_stream(prompt, request)


async def chat_non_stream(prompt: str, req: ChatCompletionRequest):
    start = time.time()
    if HAS_GENAI:
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = req.max_tokens
        config.temperature = req.temperature
        config.top_p = req.top_p
        result = pipeline.generate(prompt, config)
        text = result if isinstance(result, str) else str(result)
    elif HAS_OPTIMUM:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = pipeline.generate(
            **inputs, max_new_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p,
            do_sample=req.temperature > 0
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = text[len(prompt):]
    else:
        raise HTTPException(status_code=500, detail="No backend available")

    prompt_tokens = len(prompt) // 4
    completion_tokens = len(text) // 4
    return JSONResponse(content=make_completion_response(text, req.model, prompt_tokens, completion_tokens))


async def chat_stream_generator(prompt: str, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    if HAS_GENAI:
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = req.max_tokens
        config.temperature = req.temperature
        config.top_p = req.top_p

        def callback(subword):
            return True

        loop = asyncio.get_event_loop()

        def generate():
            pipeline.generate(prompt, config, callback)
            return ""

        result = await loop.run_in_executor(None, generate)
        # Simulated streaming: yield word-by-word
        words = prompt.split(" ")  # Placeholder; real implementation should queue tokens
        # For production, use a threading.Queue between callback and async generator
        yield make_chunk(" ", req.model)
        yield make_final_chunk(req.model)
    elif HAS_OPTIMUM:
        loop = asyncio.get_event_loop()

        def generate():
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = pipeline.generate(
                **inputs, max_new_tokens=req.max_tokens,
                temperature=req.temperature, top_p=req.top_p,
                do_sample=req.temperature > 0
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = await loop.run_in_executor(None, generate)
        result = result[len(prompt):]
        words = result.split(" ")
        for i, word in enumerate(words):
            sep = " " if i > 0 else ""
            yield make_chunk(sep + word, req.model)
            await asyncio.sleep(0.01)
        yield make_final_chunk(req.model)
    else:
        yield make_final_chunk(req.model, "error")


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if request.stream:
        return StreamingResponse(chat_stream_generator(request.prompt, request), media_type="text/event-stream")
    else:
        class FakeReq:
            model = request.model
            max_tokens = request.max_tokens
            temperature = request.temperature
            top_p = request.top_p
        return await chat_non_stream(request.prompt, FakeReq())


if __name__ == "__main__":
    print(f"[START] NIM-OV-Bridge API Server | Model: {MODEL_PATH} | Device: {DEVICE} | Port: {API_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")