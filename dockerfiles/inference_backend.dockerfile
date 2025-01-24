# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN mkdir -p app

COPY src/starfish/apis/inference_backend.py app/inference_backend.py
COPY src/starfish/model.py model.py
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
ENV PORT=8080
EXPOSE $PORT
CMD exec uvicorn app.inference_backend:app --host 0.0.0.0 --port $PORT
