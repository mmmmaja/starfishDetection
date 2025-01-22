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
# Set environment variable for PORT with a default value
ENV PORT=8080


EXPOSE $PORT
# Start the application using Gunicorn with Uvicorn workers
CMD ["gunicorn", "app.inference_backend:app", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]


# Instructions on how to build and run the image and then deploy the API to the cloud

# 1. Build the image
    # docker build -t backend:latest -f dockerfiles/inference_backend.dockerfile .

# 2. Run image
    # docker run --rm -p 8080:8080 -e "PORT=8080" backend:latest

# 3. Go to http://localhost:8080/docs to test the API

# 4. Deploy the the cloud:

    # docker tag backend:latest us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest
    # docker push us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest

    # Verify the images in the artifact registry
    # gcloud artifacts docker images list us-central1-docker.pkg.dev/starfish-detection/frontend-backend

    # gcloud run deploy backend --image=us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest --region=us-central1 --platform=managed --allow-unauthenticated --port=8080
    
