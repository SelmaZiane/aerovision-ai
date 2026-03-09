FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .
COPY specs.json .
COPY static/ static/

# Railway uses dynamic $PORT, HuggingFace Spaces uses 7860
EXPOSE 7860

# PORT env var is set by Railway; falls back to 7860 for HF Spaces
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
