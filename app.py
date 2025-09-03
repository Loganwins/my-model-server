import os
import runpod
import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.getenv("HF_TOKEN")  # set in RunPod Secrets/Env
MODEL_ID = os.getenv("MODEL_ID", "askfjhaskjgh/UbermenschetienASI")

tokenizer = None
model = None

def _load_tokenizer():
    # Try fast first; fall back to slow (SentencePiece) if repo lacks tokenizer.json
    try:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=True, token=HF_TOKEN, trust_remote_code=False
        )
        if getattr(tok, "is_fast", False):
            return tok
    except Exception:
        pass
    return AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=False, token=HF_TOKEN, trust_remote_code=False
    )

def _lazy_load():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return
    tokenizer = _load_tokenizer()
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
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)

def _generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        _lazy_load()
        inp = event.get("input") or {}
        prompt = inp.get("prompt")
        messages = inp.get("messages")
        if prompt == "__debug__":
            return {"output": {
                "model": MODEL_ID,
                "tokenizer": type(tokenizer).__name__,
                "is_fast": bool(getattr(tokenizer, "is_fast", False)),
                "has_template": bool(hasattr(tokenizer, "apply_chat_template")),
                "cuda": torch.cuda.is_available(),
            }}
        if not prompt and messages:
            prompt = _make_prompt(messages)
        if not prompt:
            return {"error": "Provide 'prompt' or 'messages'."}
        text = _generate(prompt, int(inp.get("max_new_tokens", 200)), float(inp.get("temperature", 0.7)))
        return {"output": text}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

runpod.serverless.start({"handler": handler})
