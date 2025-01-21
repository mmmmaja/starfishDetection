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

COPY src/starfish/apis/inference_backend.py inference_backend.py
COPY src/starfish/model.py model.py
COPY requirements_backend_inference.txt requirements_backend_inference.txt
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend_inference.txt.txt

# Set environment variable for PORT with a default value
ENV PORT=8080


EXPOSE $PORT
# Start the application using Gunicorn with Uvicorn workers
CMD ["gunicorn", "inference_backend:app", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]



# 1. Build the image
# docker build -t backend:latest -f dockerfiles/inference_backend.dockerfile .

# 2. Run image
# docker run --rm -p 8000:8000 -e "PORT=8000" backend:latest

# 3. Go to http://localhost:8000/docs to test the API

# 4. Deploy the the cloud:
    
    # docker tag backend:latest us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest
    # docker push us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest

    # Verify the images in the artifact registry
    # gcloud artifacts docker images list us-central1-docker.pkg.dev/starfish-detection/frontend-backend


    # gcloud run deploy backend --image=us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest --region=us-central1 --platform=managed --allow-unauthenticated



# docker tag \
#     backend:latest \
#     us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest
# docker push \
# #     us-central1.pkg.dev/starfish-detection/frontend-backend/backend:latest
# gcloud run deploy backend --image=us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest --region=us-central1 --platform=managed

# gcloud artifacts repositories create frontend-backend --repository-format=docker --location=us-central1 --description="Docker repository for frontend and backend services"

# export MYENDPOINT=$(gcloud run services describe backend --region=us-central1 --format="value(status.url)")
# curl -X 'POST' \
#     $MYENDPOINT/inference \
#     -H 'accept: application/json' \
#     -H 'Content-Type: multipart/form-data' \
#     -F 'file=@C:\\Users\\mjgoj\\Desktop\\starfishDetection\\data\\raw\\train_images\\video_0\0.jpg;type=image/jpeg'

    