import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Get Hugging Face token from RunPod secrets
hf_token = os.getenv("HF_TOKEN")

# Load your private model from Hugging Face
pipe = pipeline(
    task="text-generation",
    model="askfjhaskjgh/UbermenschetienASI",  # your Hugging Face repo name
    use_auth_token=hf_token
)

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class Request(BaseModel):
    prompt: str

# Define inference route
@app.post("/infer")
def infer(request: Request):
    try:
        output = pipe(
            request.prompt,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )[0]["generated_text"]
        return {"result": output}
    except Exception as e:
        return {"error": str(e)}


