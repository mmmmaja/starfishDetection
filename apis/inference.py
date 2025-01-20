import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse


import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import sys
from model import NMS, FasterRCNNLightning

"""
Create a FastAPI application that can do inference using the model (M22)
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle the lifespan of the FastAPI application
    """
    print("Welcome in the inference API!")
    yield
    print("Goodbye!")


# Create the FastAPI app
app = FastAPI(lifespan=lifespan)

# Load the model
# TODO: Change the path to the model once it is final
model_path = "epoch=0-step=1.ckpt"

# Load the pytorch lightning model
try:
    model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=model_path, num_classes=2)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)
    sys.exit(1)


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

    # Read the image once it was uploaded
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()

    # Preprocess the image to match the model's input requirements
    image = cv2.imread("image.jpg")
    image_processed = preprocess_image(image)
    # Add a batch dimension
    batch = image_processed.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        model.eval()
        # Prediction is the bounding boxes and the scores
        prediction = model(batch.to(device))

    # TODO: return just the JSON response
    
    # Extract the scores and boxes from the prediction
    scores = prediction[0]['scores']
    boxes = prediction[0]['boxes']

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

    cv2.imwrite("output.jpg", image)

    return FileResponse("output.jpg")


