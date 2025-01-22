import os
import sys

# Insert the parent directory into the path so that we can import the model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from model import FasterRCNNLightning
import torch
from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from fastapi.exceptions import HTTPException
import cv2
from fastapi.responses import FileResponse
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import wandb
import torchvision


# set RUNNING_LOCALLY=True

# Determine the running environment
RUNNING_LOCALLY = os.getenv("RUNNING_LOCALLY", "False").lower() in ("true", "1", "t")
print(f"Running locally: {RUNNING_LOCALLY}")


# entity = "luciagordon-harvard-university"
# project = "starfishDetection-src_starfish" q
ARTIFACT_PATH = "luciagordon-harvard-university/Starfish Detection/model-z6xxz6rn:v0"


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
    global model, device

    # Initialize W&B API only if not running locally
    if not RUNNING_LOCALLY:

        # Initialize W&B using environment variable
        key = os.getenv("WANDB_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="W&B API key not found in environment variables.")
        else:
            print("Logging in with W&B API key...")
            wandb.login(key=key)
    else:
        wandb.login()

    try:
        # Initialize W&B API
        api = wandb.Api()

        # Get the artifact
        artifact = api.artifact(ARTIFACT_PATH, type="model")
        artifact_dir = artifact.download()  # Downloads to a directory and returns the path

        model_path = os.path.join(artifact_dir, "model.ckpt")

        # Load the model
        model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=model_path, num_classes=2)

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

    keep_scores, keep_boxes = NMS(scores, boxes)
    print(f'Before NMS: {len(scores)} After NMS: {len(keep_scores)}')

    # Draw the bounding boxes on the image
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Resize the image
    image = cv2.resize(image, (640, 640))

    boxes_data = []
    for i, box in enumerate(keep_boxes):
        
        x1, y1, x2, y2 = box
        boxes_data.append({
                "score": float(keep_scores[i]),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
        
        # Add the bounding box to the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
        
        # Add the confidence score to the bounding box
        cv2.putText(
            image, text=f"{keep_scores[i]:.2f}", org=(int(x1), int(y1)), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)
    
    return image



def NMS(scores, boxes, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    :param scores: Tensor of shape (N,) containing the confidence scores
    :param boxes: Tensor of shape (N, 4) containing the predicted boxes
    """

    # 1. Sort the predictions by confidence scores
    sorted_indices = torch.argsort(scores, descending=True)

    # 2. Create a list to store the indices of the predictions to keep
    keep_indices = []

    while sorted_indices.numel() > 0:
        # Keep the prediction with the highest confidence score
        keep_indices.append(sorted_indices[0].item())

        # Calculate the IoU of the first prediction with all other predictions
        ious = torchvision.ops.box_iou(boxes[sorted_indices[0]].unsqueeze(0), boxes[sorted_indices])

        # Discard predictions with IoU greater than the threshold
        sorted_indices = sorted_indices[ious[0] <= iou_threshold]

    # Get the boxes and scores to keep
    keep_boxes = boxes[keep_indices]
    keep_scores = scores[keep_indices]
    return keep_scores, keep_boxes


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



@app.post("/show/")
# async def: Defines an asynchronous function, allowing FastAPI to handle other requests 
# while waiting for I/O operations (like reading a file) to complete.
async def show(data: UploadFile = File(...)):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)

    # Preprocess the image
    image = cv2.imread("image.jpg")
    # Preprocess the image
    image_processed = preprocess_image(image)
    # Add a batch dimension
    batch = image_processed.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        model.eval()
        # Prediction is the bounding boxes and the scores
        prediction = model(batch.to(device))

    image = process_result({"scores": prediction[0]['scores'], "boxes": prediction[0]['boxes']}, image)

    cv2.imwrite("output.jpg", image)
    return FileResponse("output.jpg")


# TODO: Tasks   
# Deploy a drift detection API to the cloud (M27)
# Instrument the API with a couple of system metrics (M28)
# For instance: The number of requests per minute.


# to run locally:
# uvicorn inference_backend:app --reload
