import os
import sys

# Insert the parent directory into the path so that we can import the model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from model import FasterRCNNLightning, NMS
import torch
from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from fastapi.exceptions import HTTPException
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import wandb
import streamlit as st


# Initialize W&B
wandb.login()


# entity = "luciagordon-harvard-university"
# project = "starfishDetection-src_starfish"
ARTIFACT_PATH = "luciagordon-harvard-university/Starfish Detection/model-4xw7155d:v0"


"""
Create a FastAPI application that can do inference using the model (M22)
This is the backend service that retrieves the prediction from the image
"""    


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle the lifespan of the FastAPI application
    """
    global model, device

    # Initialize W&B API
    api = wandb.Api()

    try:
        # Get the artifact
        artifact = api.artifact(ARTIFACT_PATH, type="model")
        artifact_dir = artifact.download()  # Downloads to a directory and returns the path

        model_path = os.path.join(artifact_dir, "model.ckpt")
        
        # model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=model_path, num_classes=2)
        model = FasterRCNNLightning(num_classes=2)
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model from W&B: {str(e)}")
        
    # Configure the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    yield
    # Clean up
    del model, device


# Create the FastAPI app
app = FastAPI(lifespan=lifespan)
#xxx



def preprocess_image(image):
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
    transform = A.Compose([
            A.Resize(640, 640),
            ToTensorV2()
        ], 
    )
    return transform(image=image)["image"]


def process_result(prediction, image):
    # Extract the scores and boxes from the prediction
    scores = prediction['scores']
    boxes = prediction['boxes']

    # Perform non-maximum suppression to remove overlapping bounding boxes
    keep_scores, keep_boxes = NMS(scores, boxes, iou_threshold=0.001)

    # Draw the bounding boxes on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(keep_boxes):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        # Add the bounding box to the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        
        # Add the confidence score to the bounding box
        score = keep_scores[i]
        cv2.putText(image, f"{score:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image


@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {"message": "Hello from the backend!"}


@app.post("/inference/")
# async def: Defines an asynchronous function, allowing FastAPI to handle other requests 
# while waiting for I/O operations (like reading a file) to complete.
async def inference(data: UploadFile = File(...)):
    """
    Perform inference on the uploaded image.
    :param data: The uploaded image file
    """

    # Read the image once it was uploaded
    with open('image.jpg', 'wb') as image:
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
            prediction = model(batch.to(device))
            print(prediction)
            return {
                "scores": prediction[0]['scores'].tolist(),
                "boxes": prediction[0]['boxes'].tolist()
            }
    except Exception as e:
        raise HTTPException(status_code=500) from e


@app.post("/show_inference/")
# async def: Defines an asynchronous function, allowing FastAPI to handle other requests 
# while waiting for I/O operations (like reading a file) to complete.
async def inference(data: UploadFile = File(...)):
    """
    Perform inference on the uploaded image.
    This is the testing version of the inference function that plots the bounding boxes on the image.
    :param data: The uploaded image file
    """

    # Read the image once it was uploaded
    with open('image.jpg', 'wb') as image:
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

    
    # Perform inference
    with torch.no_grad():
        model.eval()
        # Prediction is the bounding boxes and the scores
        prediction = model(batch.to(device))
        print(prediction)
        prediction = {
            "scores": prediction[0]['scores'].tolist(),
            "boxes": prediction[0]['boxes'].tolist()
        }
        if len(prediction["boxes"]) == 0:
            raise HTTPException(status_code=400, detail="No objects detected in the image.")
            
    try:
        processed_image = process_result(prediction, image)
        st.image(processed_image, caption="Uploaded Image")
        st.write("Predicted Bounding Boxes:", prediction["boxes"])
    except Exception as e:
        raise HTTPException(status_code=500) from e

# TODO: Tasks   
# Deploy a drift detection API to the cloud (M27)
# Instrument the API with a couple of system metrics (M28)

# to run locally:
# uvicorn inference_backend:app --reload

