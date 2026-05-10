FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SITESAFE_PORT=7860 \
    OLLAMA_HOST=http://ollama:11434

WORKDIR /app

# Build deps for some Python wheels (e.g. fpdf2) — slim image lacks them
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
        curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

COPY . .

# Build the OSHA DB at image-build time so the container starts fast.
RUN python data/build_osha_db.py

EXPOSE 7860

# Honour Gradio's expectation of binding 0.0.0.0 inside containers
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "app/sitesafe_app.py"]
