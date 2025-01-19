# Base image
FROM python:3.11-slim AS base
# FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl gnupg libgl1-mesa-glx libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
apt-get update && apt-get install -y google-cloud-sdk && \
apt-get clean && rm -rf /var/lib/apt/lists/*

COPY configs configs/
COPY src src/
COPY requirements.txt requirements.txt
# COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

# ENTRYPOINT ["sh", "-c", "gsutil -m cp -r gs://starfish-detection-data . && python -u src/starfish/train.py"]
ENTRYPOINT ["sh", "-c", "python -u src/starfish/train.py"]
