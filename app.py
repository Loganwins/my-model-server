import os
import runpod
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "askfjhaskjgh/UbermenschetienASI"

tokenizer = None
model = None

def _lazy_load():
    """
    Load tokenizer/model once, using the slow tokenizer to avoid conversion
    errors (SentencePiece/Tiktoken). Also maps to GPU if available.
    """
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return

    # Force slow tokenizer (no conversion). The 'token' arg is the modern
    # equivalent of the old 'use_auth_token'.
    tokenizer_args = dict(use_fast=False)
    if HF_TOKEN:
        tokenizer_args["token"] = HF_TOKEN

    tokenizer_local = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_args)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_args = dict(
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if HF_TOKEN:
        model_args["token"] = HF_TOKEN

    model_local = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_args)

    # Some causal models have no pad token; use eos as pad to keep generate happy.
    if tokenizer_local.pad_token_id is None and tokenizer_local.eos_token_id is not None:
        tokenizer_local.pad_token = tokenizer_local.eos_token

    # Commit to globals only after successful load
    globals()["tokenizer"] = tokenizer_local
    globals()["model"] = model_local


def _build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Prefer a model's chat template when available; otherwise fall back
    to a simple role-tagged transcript.
    """
    _lazy_load()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback
    buf = []
    for m in messages:
        buf.append(f"{m.get('role','user')}: {m.get('content','')}")
    buf.append("assistant:")
    return "\n".join(buf)


def _generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    _lazy_load()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.

    Input schema:
      event["input"] = {
        "prompt": "...",                # OR
        "messages": [{"role": "...", "content": "..."}],
        "max_new_tokens": 200,
        "temperature": 0.7
      }
    """
    try:
        inp = event.get("input") or {}
        messages = inp.get("messages")
        prompt = inp.get("prompt")

        if messages and not prompt:
            prompt = _build_prompt_from_messages(messages)
        if not prompt:
            return {"error": "Provide either 'prompt' or 'messages' in input."}

        max_new_tokens = int(inp.get("max_new_tokens", 200))
        temperature = float(inp.get("temperature", 0.7))

        out_text = _generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return {"output": out_text}

    except Exception as e:
        # Bubble the full error string back so you can see it in Requests logs
        return {"error": f"{type(e).__name__}: {e}"}


# Start the RunPod event loop
runpod.serverless.start({"handler": handler})
