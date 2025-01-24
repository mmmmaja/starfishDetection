import os
import sys

# Insert the parent directory into the path so that we can import the model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

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
from model import FasterRCNNLightning

"""
Create a FastAPI application that can do inference using the model (M22)
This is the backend service that retrieves the prediction from the image
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle the lifespan of the FastAPI application
    :param app: The FastAPI application
    """
    global model, onnx_session, device

    model_path = "/gcs/starfish-model/model.ckpt"
    local_onnx_path = "/gcs/faster-rcnn-onnx/FasterRCNN.onnx"

    try:
        model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=model_path, num_classes=2) # load torch model
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        onnx_session = rt.InferenceSession(local_onnx_path, providers=providers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Configure the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    yield
    # Clean up
    del model, onnx_session, device


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
# async def: Defines an asynchronous function, allowing FastAPI to handle other requests
# while waiting for I/O operations (like reading a file) to complete.
async def inference(data: UploadFile = File(...)) -> dict:
    """
    Perform inference on the uploaded image.
    :param data: The uploaded image file
    :return: The prediction from the model
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
    batch = image_processed.unsqueeze(0)

    try:
        # Perform inference
        with torch.no_grad():
            model.eval()
            # Prediction is the bounding boxes and the scores
            prediction = model.model(batch.to(device))
            return {"scores": prediction[0]["scores"].tolist(), "boxes": prediction[0]["boxes"].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500) from e


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
