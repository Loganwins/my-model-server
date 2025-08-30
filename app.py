import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ---------------------------
# Load model + tokenizer
# ---------------------------
MODEL = "askfjhaskjgh/UbermenschetienASI"  # your HF repo with weights
print(f"Loading model: {MODEL}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# ---------------------------
# Define handler
# ---------------------------
def handler(job):
    """
    job["input"] must be in the form:
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }
    """
    messages = job["input"].get("messages", [])
    if not messages:
        return {"error": "No messages provided."}

    # Build conversation history
    conversation = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation += f"{role}: {content}\n"

    # Add assistant turn
    conversation += "assistant:"

    # Generate response
    outputs = pipe(
        conversation,
        max_length=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )

    reply = outputs[0]["generated_text"].split("assistant:")[-1].strip()
    return {"reply": reply}

# ---------------------------
# Start RunPod Serverless
# ---------------------------
runpod.serverless.start({"handler": handler})
