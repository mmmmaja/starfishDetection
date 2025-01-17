import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from http import HTTPStatus
import base64
from contextlib import asynccontextmanager
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Inser the path to the main directory (1 level up)
parent_directory = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_directory))

from src.starfish.model import NMS, FasterRCNNLightning

"""
Create a FastAPI application that can do inference using the model (M22)
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Hello!")
    yield
    print("Goodbye!")

app = FastAPI(lifespan=lifespan)

# Load the model
model_path = parent_directory / "lightning_logs" / "version_0" / "checkpoints" / "epoch=0-step=1.ckpt"
# Make sure the checkpoint file exists
if not model_path.is_file():
    raise FileNotFoundError(f"Checkpoint file not found at {model_path}")

# Load the pytorch lightning model
model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=model_path, num_classes=2)
print("Model loaded successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


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
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ], 
    )
    return transform(image=image)["image"]


@app.post("/inference/")
# async def: Defines an asynchronous function, allowing FastAPI to handle other requests 
# while waiting for I/O operations (like reading a file) to complete.
async def inference(data: UploadFile = File(...)):
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
    
    scores = prediction[0]['scores']
    boxes = prediction[0]['boxes']

    keep_scores, keep_boxes = NMS(scores, boxes)

    # Draw the bounding boxes on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    boxes_data = []
    for i, box in enumerate(keep_boxes):
        
        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height

        boxes_data.append({
                "score": float(keep_scores[i]),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
        
        # Add the bounding box to the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        
        # Add the confidence score to the bounding box
        cv2.putText(
            image, text=f"{keep_scores[i]:.2f}", org=(int(x1), int(y1)), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

    # cv2.imwrite("output.jpg", image)

    # return FileResponse("output.jpg")

     # Encode image to JPEG format in memory
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image.")

        # Convert to base64 for JSON serialization
        image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

        # Prepare JSON response
        response = {
            "image": image_base64,
            "boxes": boxes_data,
            "message": "Inference successful.",
            "status_code": HTTPStatus.OK.value
        }

        return JSONResponse(content=response, status_code=HTTPStatus.OK)


