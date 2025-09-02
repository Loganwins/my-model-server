import os
import json
import runpod
from typing import Any, Dict, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    __version__ as transformers_version,
)
from huggingface_hub import login, list_repo_files, __version__ as hub_version

# ---- Config -----------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "askfjhaskjgh/UbermenschetienASI")

# Read HF token from either name (we set BOTH in RunPod):
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Keep cache paths stable inside the container
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/transformers")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster downloads

# -----------------------------------------------------------------------------
tokenizer = None
model = None


def _ensure_hf_login() -> None:
    """Make sure the hub knows our token (no-op if already ok)."""
    if not HF_TOKEN:
        raise RuntimeError("HF token is missing; set HF_TOKEN / HUGGINGFACE_HUB_TOKEN in RunPod.")
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception:
        # best-effort; even if login() fails, passing token=... still works
        pass


def _lazy_load() -> None:
    """Load tokenizer and model once, with robust fallbacks."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return

    _ensure_hf_login()

    # Quick permission check (gives clearer error than from_pretrained)
    try:
        _ = list_repo_files(MODEL_ID, token=HF_TOKEN)
    except Exception as e:
        raise RuntimeError(f"HF auth/list failed for '{MODEL_ID}': {e}")

    # 1) Tokenizer: try fast first, then slow (SentencePiece) if needed
    tok_errs = []
    for use_fast in (True, False):
        try:
            tokenizer_candidate = AutoTokenizer.from_pretrained(
                MODEL_ID,
                token=HF_TOKEN,          # new param name (works on recent transformers)
                use_fast=use_fast,
                trust_remote_code=True,  # safe here; your repo
            )
            tokenizer = tokenizer_candidate
            break
        except Exception as e:
            tok_errs.append(f"use_fast={use_fast}: {e}")
            tokenizer_candidate = None

    if tokenizer is None:
        raise RuntimeError(
            "Failed to load tokenizer. Tried fast and slow.\n" + "\n".join(tok_errs)
        )

    # 2) Model
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model_candidate = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model_candidate.eval()
        model = model_candidate
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{MODEL_ID}': {e}")


def _build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """Use chat template when available; otherwise simple fallback."""
    _lazy_load()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    # fallback
    lines = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def _generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    _lazy_load()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    # return ONLY the newly generated text (strip the prompt)
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler. Accepts:
      {
        "input": {
          "prompt": "...",                        # or
          "messages": [{"role":"user","content":"..."}],
          "max_new_tokens": 200,
          "temperature": 0.7
        }
      }
    """
    try:
        inp = (event or {}).get("input", {}) or {}
        prompt = inp.get("prompt")
        messages = inp.get("messages")

        # Debug hook to surface environment quickly from curl:
        if prompt == "__debug__":
            return {
                "model_id": MODEL_ID,
                "has_token": bool(HF_TOKEN),
                "cuda": torch.cuda.is_available(),
                "transformers": transformers_version,
                "hub": hub_version,
                "env_seen": {
                    k: os.getenv(k)
                    for k in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME", "TRANSFORMERS_CACHE"]
                },
            }

        if not prompt and messages:
            prompt = _build_prompt_from_messages(messages)
        if not prompt:
            return {"error": "Provide either 'prompt' or 'messages' in input."}

        max_new_tokens = int(inp.get("max_new_tokens", 200))
        temperature = float(inp.get("temperature", 0.7))

        text = _generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return {"output": text}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# Start RunPod worker
runpod.serverless.start({"handler": handler})
