# One image for the API and the UI (docker-compose.yml runs it twice with different commands).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/models

# Tesseract for OCR of scanned pages
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user; data, index, outputs and model cache are mounted volumes.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models /app/index /app/outputs \
    && chown -R app /app /models
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
