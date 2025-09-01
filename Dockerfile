# ---- Base image: PyTorch + CUDA on Ubuntu 22.04 ----
FROM runpod/pytorch:2.3.1-py3.10-cuda12.1.1

# Workdir
WORKDIR /app

# (Optional) lightweight tools
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer cache)
COPY requirements.txt .

# Install deps (keep torch as-is; base already has CUDA torch)
# The extra-index ensures any torch re-install stays on CUDA wheels if needed.
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu121

# Copy app code
COPY . .

# Useful envs
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HOME=/root/.cache/huggingface

# Start the RunPod worker (Queue endpoints)
CMD ["python", "app.py"]
