from fastapi.testclient import TestClient
from starfish.apis.inference_backend import app
import pytest
import os
import requests
import numpy as np
import cv2

client = TestClient(app)

def test_backend():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello from the backend!"}

@pytest.mark.asyncio
async def test_inference_endpoint():
    # Create a temporary image file for testing
    image_path = "test_image.jpg"
    random_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(image_path, random_image)

    with open(image_path, "rb") as image_file:
        response = requests.post("https://backend-638730968773.us-central1.run.app/inference/", files={"data": image_file})

    assert response.status_code == 200
    assert "scores" in response.json()
    assert "boxes" in response.json()

    os.remove(image_path)

def test_frontend():
    url = "https://frontend-638730968773.us-central1.run.app"
    response = requests.get(url)
    
    assert response.status_code == 200, f"Failed to load page: {response.status_code}"
