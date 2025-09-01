FROM python:3.10-slim

WORKDIR /app

# minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CUDA 12.1 build of torch (GPU) first
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Queue worker (not a web server)
CMD ["python", "app.py"]
