import os

# 🔒 Force Transformers to never use fast tokenizers anywhere in this process.
# Do this before importing transformers.
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"

import runpod
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "askfjhaskjgh/UbermenschetienASI"

tokenizer = None
model = None


def _lazy_load():
    """
    Load tokenizer/model once. We explicitly disable fast tokenizers, so no
    SentencePiece/Tiktoken conversion is attempted.
    """
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return

    tok_kwargs = dict(use_fast=False)
    if HF_TOKEN:
        tok_kwargs["token"] = HF_TOKEN

    print("[boot] loading tokenizer (slow)…")
    tokenizer_local = AutoTokenizer.from_pretrained(MODEL_ID, **tok_kwargs)
    print(f"[boot] tokenizer loaded. is_fast={getattr(tokenizer_local, 'is_fast', None)}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mdl_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN

    print(f"[boot] loading model ({dtype})…")
    model_local = AutoModelForCausalLM.from_pretrained(MODEL_ID, **mdl_kwargs)
    print("[boot] model loaded.")

    # Some causal models have no pad token; use eos as pad to keep generate() happy.
    if tokenizer_local.pad_token_id is None and tokenizer_local.eos_token_id is not None:
        tokenizer_local.pad_token = tokenizer_local.eos_token

    # Publish to globals only after successful loads
    globals()["tokenizer"] = tokenizer_local
    globals()["model"] = model_local


def _build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Prefer chat template; fall back to a simple transcript.
    """
    _lazy_load()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback
    parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)


def _generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    _lazy_load()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.

    Input:
      {
        "prompt": "...",                     # OR
        "messages": [{"role":"user","content":"..."}],
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

        max_new = int(inp.get("max_new_tokens", 200))
        temp = float(inp.get("temperature", 0.7))

        text = _generate(prompt, max_new_tokens=max_new, temperature=temp)
        return {"output": text}

    except Exception as e:
        # Return full error text so it shows in Requests tab
        return {"error": f"{type(e).__name__}: {e}"}


# Start RunPod worker loop
runpod.serverless.start({"handler": handler})
