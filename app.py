import os
import runpod
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.getenv("HF_TOKEN")  # (optional) set in RunPod secrets
MODEL_ID = "askfjhaskjgh/UbermenschetienASI"

tokenizer = None
model = None


def _lazy_load():
    """Load tokenizer/model once, on first request."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return

    print(f"🔧 Loading model: {MODEL_ID}")
    # Use the *slow* tokenizer (avoids SP/Tiktoken converter issues).
    # Newer Transformers uses `token=`, older uses `use_auth_token=`.
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=False)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, use_fast=False)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            use_auth_token=HF_TOKEN,
            torch_dtype=dtype,
            device_map="auto",
        )

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model loaded.")


def _build_prompt(messages: List[Dict[str, str]] | None, prompt: str | None) -> str:
    """Use chat template if available; else simple role-tagged history."""
    _lazy_load()
    if messages and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if messages:
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text += f"{role}: {content}\n"
        return text + "assistant:"
    if prompt:
        return prompt
    raise ValueError("Provide either 'messages' or 'prompt'.")


def generate_text(prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
    """
    Thin wrapper over model.generate().
    - prompt: full prompt string
    - returns decoded text
    """
    _lazy_load()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects event['input'] like:
    {
      "prompt": "...",  // OR
      "messages": [{"role":"user","content":"..."}, ...],
      "max_new_tokens": 200,
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": 50
    }
    """
    try:
        inp = event.get("input") or {}
        prompt_in = inp.get("prompt")
        messages = inp.get("messages")

        max_new_tokens = int(inp.get("max_new_tokens", 200))
        temperature = float(inp.get("temperature", 0.7))
        top_p = float(inp.get("top_p", 0.9))
        top_k = int(inp.get("top_k", 50))

        built = _build_prompt(messages, prompt_in)
        text = generate_text(built, max_new_tokens, temperature, top_p, top_k)

        # If we used a messages transcript, return only the assistant’s last turn (best-effort split)
        if messages:
            after = text.split("assistant:")[-1].strip()
            return {"reply": after, "full_text": text}

        return {"output": text}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ {err}")
        return {"error": err}


runpod.serverless.start({"handler": handler})
