FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY apis/requirements_frontend_inference.txt requirements_frontend_inference.txt
COPY apis/inference_frontend.py inference_frontend.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r app/requirements_frontend.txt

EXPOSE $PORT

CMD ["streamlit", "run", "frontend.py", "--server.port", "$PORT"]