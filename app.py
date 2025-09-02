import os
import runpod
from typing import List, Dict, Any

import torch
import transformers
import tokenizers
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "askfjhaskjgh/UbermenschetienASI"

tokenizer = None
model = None

def _boot_banner():
    print("=== [boot] ===")
    print(f"[boot] transformers: {transformers.__version__}")
    print(f"[boot] tokenizers  : {tokenizers.__version__}")
    print(f"[boot] CUDA avail? : {torch.cuda.is_available()}")
    print(f"[boot] ENV TRANSFORMERS_NO_FAST_TOKENIZER = {os.getenv('TRANSFORMERS_NO_FAST_TOKENIZER')}")
    print("================")

def _lazy_load():
    """Load the FAST tokenizer (uses tokenizer.json) and the model once."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return

    _boot_banner()

    tok_kwargs = dict(use_fast=True)  # <- force FAST
    if HF_TOKEN:
        tok_kwargs["token"] = HF_TOKEN

    print("[boot] loading FAST tokenizer…")
    t = AutoTokenizer.from_pretrained(MODEL_ID, **tok_kwargs)
    print(f"[boot] tokenizer loaded. is_fast={getattr(t, 'is_fast', None)}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mdl_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN

    print(f"[boot] loading model (dtype={dtype})…")
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, **mdl_kwargs)
    print("[boot] model loaded.")

    if t.pad_token_id is None and t.eos_token_id is not None:
        t.pad_token = t.eos_token

    tokenizer, model = t, m

def _build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    _lazy_load()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
    try:
        inp = event.get("input") or {}
        prompt = inp.get("prompt")
        msgs = inp.get("messages")

        # cheap health/debug hooks:
        if (prompt or "").strip() == "__debug__":
            _lazy_load()
            return {
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
                "is_fast": getattr(tokenizer, "is_fast", None),
                "cuda": torch.cuda.is_available()
            }

        if msgs and not prompt:
            prompt = _build_prompt_from_messages(msgs)
        if not prompt:
            return {"error": "Provide either 'prompt' or 'messages' in input."}

        max_new = int(inp.get("max_new_tokens", 200))
        temp = float(inp.get("temperature", 0.7))
        text = _generate(prompt, max_new_tokens=max_new, temperature=temp)
        return {"output": text}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

runpod.serverless.start({"handler": handler})
