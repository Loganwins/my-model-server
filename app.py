import os
import runpod
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "askfjhaskjgh/UbermenschetienASI"

tokenizer = None
model = None

def _lazy_load():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
        # Use half-precision when possible and move to GPU if available
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            use_auth_token=HF_TOKEN,
        )

def _build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Uses the tokenizer's chat template if available, otherwise falls back
    to a simple role-tagged transcript.
    """
    _lazy_load()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    # Fallback: simple concatenation
    buf = []
    for m in messages:
        buf.append(f"{m.get('role','user')}: {m.get('content','')}")
    buf.append("assistant:")
    return "\n".join(buf)

def generate_text(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    _lazy_load()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler. Expects:
      event["input"] = {
        "prompt": "...",           # OR
        "messages": [ {"role":"user","content":"..."} , ... ],
        "max_new_tokens": 200,
        "temperature": 0.7
      }
    """
    try:
        inp = event.get("input", {}) or {}
        messages = inp.get("messages")
        prompt = inp.get("prompt")

        if messages and not prompt:
            prompt = _build_prompt_from_messages(messages)
        if not prompt:
            return {"error": "Provide either 'prompt' or 'messages' in input."}

        max_new_tokens = int(inp.get("max_new_tokens", 200))
        temperature = float(inp.get("temperature", 0.7))

        text = generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return {"output": text}

    except Exception as e:
        return {"error": str(e)}

# Start RunPod worker loop
runpod.serverless.start({"handler": handler})

