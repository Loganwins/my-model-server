FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Basic OS + clean layer size
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Helpful env for HF + PyTorch
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TORCH_CUDA_ARCH_LIST="All" \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# RunPod serverless: app.py starts the handler
CMD ["python", "app.py"]
