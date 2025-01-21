# import os
# import unittest
# from fastapi.testclient import TestClient
# from apis import inference  # Replace 'main' with the filename where the FastAPI app is defined

# class TestInferenceEndpoint(unittest.TestCase):
#     def setUp(self):
#         """
#         Set up the test client and any required resources.
#         """
#         self.client = TestClient(app)
#         self.test_image_path = "test_image.jpg"  # Replace with the path to a valid test image

#         # Ensure the test image exists
#         if not os.path.exists(self.test_image_path):
#             raise FileNotFoundError(f"Test image not found at {self.test_image_path}")

#     def test_inference(self):
#         """
#         Test the inference endpoint with a valid image.
#         """
#         with open(self.test_image_path, "rb") as test_image:
#             files = {"data": ("test_image.jpg", test_image, "image/jpeg")}
#             response = self.client.post("/inference/", files=files)

#         # Assert the response contains the output image
#         output_path = "output.jpg"  # Path where the output image is saved
#         self.assertTrue(os.path.exists(output_path), "Output image not generated.")

#         # Clean up
#         if os.path.exists(output_path):
#             os.remove(output_path)

# if __name__ == "__main__":
#     unittest.main()
