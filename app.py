import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Get Hugging Face token from RunPod secrets
hf_token = os.getenv("HF_TOKEN")

# Load your private model from Hugging Face
pipe = pipeline(
    "text-generation",
    model="askfjhaskjgh/UbermenschetienASI",  # your HF repo name
    use_auth_token=hf_token
)

app = FastAPI()

class Request(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
def infer(request: Request):
    output = pipe(request.prompt, max_length=200)[0]["generated_text"]
    return {"result": output}


