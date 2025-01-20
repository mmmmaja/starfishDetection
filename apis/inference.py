import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import sys
from pathlib import Path

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
# Load the pytorch lightning model
model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=model_path, num_classes=2)
print("Model loaded successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


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
    transform = A.Compose(
        [A.Resize(640, 640), A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2), ToTensorV2()],
    )
    return transform(image=image)["image"]


@app.post("/inference/")
# async def: Defines an asynchronous function, allowing FastAPI to handle other requests
# while waiting for I/O operations (like reading a file) to complete.
async def inference(data: UploadFile = File(...)):
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    # Preprocess the image
    image = cv2.imread("image.jpg")
    # Preprocess the image
    image_processed = preprocess_image(image)
    # Add a batch dimension
    batch = image_processed.unsqueeze(0)
    print("batch shape:", batch.shape)

    # Perform inference
    with torch.no_grad():
        model.eval()
        # Prediction is the bounding boxes and the scores
        prediction = model(batch.to(device))

    scores = prediction[0]["scores"]
    boxes = prediction[0]["boxes"]

    keep_scores, keep_boxes = NMS(scores, boxes)

    # Draw the bounding boxes on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(keep_boxes):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        # Add the bounding box to the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)

        # Add the confidence score to the bounding box
        score = keep_scores[i]
        # TODO: Change the font size and thickness

    cv2.imwrite("output.jpg", image)

    return FileResponse("output.jpg")
