from google.cloud import run_v2
import streamlit as st
import os
import requests
from starfish.apis.inference_backend import *


PROJECT = "starfish-detection"
REGION = "us-central1"

@st.cache_resource  
def get_backend_url():
    """
    Get the URL of the backend service.
    """
    parent = f"projects/{PROJECT}/locations/{REGION}"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)

    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
        
    name = os.environ.get("BACKEND", None)
    return name


def object_detection(image, backend):
    """
    Send the image to the backend for inference
    """
    predict_url = f"{backend}/inference"
    response = requests.post(predict_url, files={"image": image}, timeout=25)
    if response.status_code == 200:
        return response.json()
    return None


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


def main():
    """
    Main function of the Streamlit frontend.
    """
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)
    
    st.title("Starfish Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = object_detection(image, backend=backend)

        if result is not None:
            
            # Process the result
            processed_image = process_result(result, image)

            # show the image and prediction
            st.image(processed_image, caption="Uploaded Image")
            st.write("Predicted Bounding Boxes:", result["boxes"])

            # Make a histogram of the scores
            data = result['scores']
            st.histogram(data, bins=10, x_label="Confidence Score", y_label="Frequency")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()


# TODO: Wandb download model