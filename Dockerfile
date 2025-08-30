FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No need to EXPOSE for queue workers, but harmless if present
EXPOSE 8000

# Start the RunPod worker (NOT a web server)
CMD ["python", "app.py"]
