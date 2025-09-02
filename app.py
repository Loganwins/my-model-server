# app.py — minimal, chat-capable, no extra deps
import os
import runpod
import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.getenv("HF_TOKEN")  # set this in RunPod env/secrets
MODEL_ID = "askfjhaskjgh/UbermenschetienASI"

tokenizer = None
model = None

def _lazy_load():
    """Load tokenizer/model once, on first request."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return

    # Fast tokenizer; requires tokenizer.json in your HF repo (you uploaded it)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        token=HF_TOKEN,
        trust_remote_code=False,
    )
    # Ensure a pad token exists
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
        trust_remote_code=False,
    )
    model.eval()

def _make_prompt(messages: List[Dict[str, str]]) -> str:
    """Use chat template when available; fallback to simple transcript."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback formatting
    buf = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        buf.append(f"{role}: {content}")
    buf.append("assistant:")
    return "\n".join(buf)

def _generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects:
      {
        "input": {
          "prompt": "text",            # OR
          "messages": [ {"role":"user","content":"..."} , ... ],
          "max_new_tokens": 200,
          "temperature": 0.7
        }
      }
    """
    try:
        _lazy_load()

        inp = event.get("input") or {}
        prompt = inp.get("prompt")
        messages = inp.get("messages")

        # simple debug hook
        if prompt == "__debug__":
            return {
                "output": {
                    "model": MODEL_ID,
                    "tokenizer_fast": type(tokenizer).__name__,
                    "has_template": bool(hasattr(tokenizer, "apply_chat_template")),
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "cuda": torch.cuda.is_available(),
                }
            }

        if not prompt and messages:
            prompt = _make_prompt(messages)

        if not prompt:
            return {"error": "Provide 'prompt' or 'messages'."}

        max_new_tokens = int(inp.get("max_new_tokens", 200))
        temperature = float(inp.get("temperature", 0.7))

        text = _generate(prompt, max_new_tokens, temperature)
        return {"output": text}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

runpod.serverless.start({"handler": handler})
