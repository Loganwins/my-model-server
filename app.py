import os
import runpod
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------- Config --------
MODEL_ID = "askfjhaskjgh/UbermenschetienASI"     # your HF repo
HF_TOKEN = os.getenv("HF_TOKEN")                 # set in RunPod -> Endpoint -> Secrets

# Generation defaults (tweak as you like)
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE    = 0.7
DEFAULT_TOP_P          = 0.9
DEFAULT_TOP_K          = 50
DEFAULT_REP_PENALTY    = 1.05

# Globals (lazy-loaded)
tokenizer = None
model = None


def _auth_kwargs() -> Dict[str, Any]:
    """Support both new (token=) and old (use_auth_token=) HF auth args."""
    try:
        # new style
        return {"token": HF_TOKEN} if HF_TOKEN else {}
    except TypeError:
        # fall back if transformers is older
        return {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}


def _lazy_load() -> None:
    """Load tokenizer/model once on first request."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return

    print(f"🔧 Loading model: {MODEL_ID}")
    auth = _auth_kwargs()

    # Use slow tokenizer to avoid SP/Tiktoken converter issues.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=False, **auth
        )
    except TypeError:
        # very old transformers fallback
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **auth)

    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        **auth,
    )

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model + tokenizer ready.")


def _build_prompt(messages: Optional[List[Dict[str, str]]], prompt: Optional[str]) -> str:
    """Prefer chat template; else simple role-tagged history; else raw prompt."""
    _lazy_load()

    if messages:
        # Try tokenizer's chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: role-tag transcript + assistant cue
        parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
        return "\n".join(parts) + "\nassistant:"

    if prompt:
        return prompt

    raise ValueError("Provide either 'messages' or 'prompt'.")


def _generate(
    built_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    """Thin wrapper around model.generate -> decoded text."""
    _lazy_load()

    inputs = tokenizer(built_prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input (either):
      {
        "prompt": "...",
        "max_new_tokens": 200, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.05
      }
    or:
      {
        "messages": [{"role":"user","content":"..."}, ...],
        "max_new_tokens": 200, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.05
      }
    """
    try:
        inp = event.get("input") or {}

        prompt_in  = inp.get("prompt")
        messages   = inp.get("messages")

        max_new    = int(inp.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        temperature= float(inp.get("temperature",    DEFAULT_TEMPERATURE))
        top_p      = float(inp.get("top_p",          DEFAULT_TOP_P))
        top_k      = int(inp.get("top_k",            DEFAULT_TOP_K))
        rep_pen    = float(inp.get("repetition_penalty", DEFAULT_REP_PENALTY))

        built = _build_prompt(messages, prompt_in)
        text  = _generate(built, max_new, temperature, top_p, top_k, rep_pen)

        # If chat input, try to return just the assistant's last turn, but include full text too
        if messages:
            reply = text.split("assistant:")[-1].strip() if "assistant:" in text else text
            return {"reply": reply, "full_text": text}

        # Plain prompt
        return {"output": text}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ {err}")
        return {"error": err}


# Start RunPod worker loop (Queue endpoints)
runpod.serverless.start({"handler": handler})
