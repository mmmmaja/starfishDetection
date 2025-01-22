from fastapi.testclient import TestClient
from app.main import app

"""
Test APIs written in FastAPI
"""

client = TestClient(app)

def test_inference(model):
    """
    Test the inference API
    TODO
    """
    url = "/inference"
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MNIST model inference API!"}