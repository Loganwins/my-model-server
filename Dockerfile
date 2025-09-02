# CUDA + PyTorch base from RunPod
FROM runpod/pytorch:3.10-2.1.2-ubuntu22.04

WORKDIR /app

# Speed up installs, keep caches predictable
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY . .

# Launch the RunPod serverless worker
CMD ["python", "-u", "app.py"]
