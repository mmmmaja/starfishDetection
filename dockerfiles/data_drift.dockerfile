FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pandas==2.2.3 fastapi==0.115.6 numpy==2.2.1
RUN pip install pillow==11.1.0 opencv-python==4.11.0.86 uvicorn==0.34.0
RUN pip install evidently==0.5.1 anyio==4.8.0 google-cloud-storage==2.19.0

COPY src/starfish/apis/data_drift.py data_drift.py

ENV PORT=8080
EXPOSE $PORT

CMD exec uvicorn data_drift:app --port $PORT --host 0.0.0.0


# docker build -t data_drift:latest -f dockerfiles/data_drift.dockerfile .
# docker run --rm -p 8000:800 -e "PORT=8000" data_drift:latest
