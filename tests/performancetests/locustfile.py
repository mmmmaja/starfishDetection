import os

import cv2
import numpy as np
from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def post_inference(self) -> None:
        """A task that simulates a user sending a POST request to the inference endpoint of the FastAPI app."""
        image_path = "test_image.jpg"
        random_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(image_path, random_image)

        with open(image_path, "rb") as image_file:
            self.client.post("/inference/", files={"data": image_file})

        if os.path.exists(image_path):
            os.remove(image_path)
