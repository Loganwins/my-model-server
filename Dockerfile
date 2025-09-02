# CUDA + PyTorch (official) runtime with Python 3.10
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps (git is handy for some hubs)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Python deps
# - transformers, accelerate, safetensors: core HF stack
# - tiktoken, sentencepiece: avoid tokenizer fallback errors
# - runpod: serverless worker loop
# - fastapi/uvicorn only needed if you also expose HTTP for local tests
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir \
      "transformers>=4.41" \
      "accelerate>=0.31" \
      "safetensors" \
      "tiktoken" \
      "sentencepiece" \
      "einops" \
      "huggingface_hub[cli]" \
      "runpod" \
      "fastapi" "uvicorn"

# Copy app
COPY app.py /app/app.py

# (Optional) keep HF cache predictable
ENV HF_HOME=/root/.cache/huggingface

# Start the RunPod serverless worker
CMD ["python", "app.py"]
