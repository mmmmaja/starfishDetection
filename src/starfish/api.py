from fastapi import FastAPI, UploadFile
from model import FasterRCNNLightning
from PIL import Image
import torch
import glob

app = FastAPI()
checkpoint_path = glob.glob("Starfish Detection/**/*.ckpt", recursive=True)[0]
model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path, num_classes=2)
model.eval()  # Set the model to evaluation mode
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the starfish detection model inference API!"}

@app.post("/predict/")
def predict(image: UploadFile):
    """Endpoint for making predictions."""
    # Load and preprocess the image
    image_data = Image.open(image.file).convert("RGB")
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # Add any other transformations here
    # ])
    # image_data = transform(image_data)
    image_data = image_data.unsqueeze(0).to(model.device)

    # Make prediction
    with torch.no_grad():
        prediction = model(image_data)

    # Process the prediction and return the result
    return {"prediction": prediction}