import os
import sys

# Insert the parent directory into the path so that we can import the model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import os
import uuid
from contextlib import asynccontextmanager

import albumentations as A
import cv2
import numpy as np
import onnxruntime as rt
import requests
import torch
from albumentations.pytorch.transforms import ToTensorV2
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from google.cloud import storage
from model import FasterRCNNLightning

"""
Create a FastAPI application that can do inference using the model (M22)
This is the backend service that retrieves the prediction from the image
"""

GCS_BUCKET_NAME = "inference_api_data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle the lifespan of the FastAPI application
    :param app: The FastAPI application
    """
    global model, onnx_session, device, storage_client, bucket

    model_path = "/gcs/starfish-model/model.ckpt"
    local_onnx_path = "/gcs/faster-rcnn-onnx/FasterRCNN.onnx"

    try:
        model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=model_path, num_classes=2)  # load torch model
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        onnx_session = rt.InferenceSession(local_onnx_path, providers=providers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    print("Model loaded successfully.")

    # Configure the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Intialize the target bucket for saving new data
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        if not bucket.exists():
            raise HTTPException(status_code=500, detail=f"GCS bucket '{GCS_BUCKET_NAME}' does not exist.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize GCS client: {str(e)}")
    print("GCS client initialized successfully.")

    yield
    # Clean up
    del model, onnx_session, device, storage_client, bucket


# Create the FastAPI app
app = FastAPI(lifespan=lifespan)


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess the image before passing it to the model
    :param image: The input image
    :return: The preprocessed image as a PyTorch tensor
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Make the range of the image between 0 and 1
    image = image / 255.0
    # Make it of a type double
    image = image.astype("float32")
    transform = A.Compose(
        [A.Resize(800, 800), ToTensorV2()],
    )
    return transform(image=image)["image"]


# Define the root endpoint
@app.get("/")
async def root() -> dict:
    """
    Root endpoint.
    """
    return {"message": "Hello from the backend!"}


@app.post("/inference/")
async def inference(data: UploadFile = File(...)) -> dict:
    """
    Perform inference on the uploaded image.
    :param data: The uploaded image file
    :return: The prediction from the model
    """
    # Validate the uploaded file type
    if not data.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read the image content
        image_bytes = await data.read()
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoding failed.")

        # Preprocess the image
        image_processed = preprocess_image(image)
        # Add a batch dimension
        batch = image_processed.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            model.eval()
            prediction = model.model(batch.to(device))

        # Upload the image to GCS
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(data.filename)[1] or ".jpg"  # Default to .jpg if no extension
        gcs_filename = f"uploaded_images/{unique_id}{file_extension}"
        blob = bucket.blob(gcs_filename)
        blob.upload_from_string(image_bytes, content_type=data.content_type)

        print(f"Uploaded image to {gcs_filename}")

        return {
            "scores": prediction[0]["scores"].tolist(),
            "boxes": prediction[0]["boxes"].tolist()
            #"image_url": blob.public_url,  # If you made the blob public
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/onnx-inference/")
async def onnx_inference(data: UploadFile = File(...)) -> dict:
    """
    Perform inference using the ONNX model.
    :param data: The uploaded image file
    :return: The prediction from the ONNX model
    """

    # Read the image once it was uploaded
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    # Preprocess the image to match the model's input requirements
    image = cv2.imread("image.jpg")
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    image_processed = preprocess_image(image)
    # Add a batch dimension
    batch = image_processed.unsqueeze(0).numpy()

    try:
        # Perform inference
        input_name = onnx_session.get_inputs()[0].name
        prediction = onnx_session.run(None, {input_name: batch})

        return {"scores": prediction[2].tolist(), "boxes": prediction[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {str(e)}")
