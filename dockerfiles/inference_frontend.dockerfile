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


COPY src/starfish/apis/inference_frontend.py inference_frontend.py
COPY requirements_frontend_inference.txt requirements_frontend_inference.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend_inference.txt
ENV PORT=8080

EXPOSE $PORT

CMD ["streamlit", "run", "inference_frontend.py", "--bind", "0.0.0.0:8080"]
