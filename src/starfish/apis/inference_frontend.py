import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import torch
import torchvision
from requests.exceptions import RequestException, Timeout

# Constants for the Google Cloud project and region
PROJECT = "starfish-detection"
REGION = "us-central1"


def NMS(scores: torch.Tensor, boxes: torch.Tensor, iou_threshold: float = 0.02) -> tuple:
    """
    Non-Maximum Suppression
    :param scores: Tensor of shape (N,) containing the confidence scores
    :param boxes: Tensor of shape (N, 4) containing the predicted boxes
    :param iou_threshold: The threshold for IoU (Intersection over Union)
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


def process_result(prediction: dict, image: np.ndarray, NMS_threshold: float = 0.02) -> np.ndarray:
    """
    Process the prediction and draw the bounding boxes on the image
    :param prediction: The prediction from the model
    :param image: The input image
    :param NMS_threshold: The threshold for Non-Maximum Suppression
    :return: The image with the bounding boxes drawn on it
    """

    # Extract the scores and boxes from the prediction
    scores = prediction["scores"]
    boxes = prediction["boxes"]

    # Run Non-Maximum Suppression (NMS) to remove overlapping boxes
    keep_scores, keep_boxes = NMS(scores, boxes, iou_threshold=NMS_threshold)
    print(f"Before NMS: {len(scores)} After NMS: {len(keep_scores)}")

    # Resize the image
    image = cv2.resize(image, (640, 640))

    # Draw the bounding boxes on the image
    boxes_data = []
    for i, box in enumerate(keep_boxes):
        x1, y1, x2, y2 = box
        boxes_data.append({"score": float(keep_scores[i]), "box": [int(x1), int(y1), int(x2), int(y2)]})

        # Add the bounding box to the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=3)

        # Add the confidence score to the bounding box
        cv2.putText(
            image,
            text=f"{keep_scores[i]:.2f}",
            org=(int(x1), int(y1)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=2,
        )

    return image


def object_detection(image: bytes, backend: str) -> dict:
    """
    Send the image to the backend for inference
    :param image: The image to send
    :param backend: The URL of the backend service
    :return: The prediction from the backend
    """
    predict_url = f"{backend}/inference/"

    try:
        response = requests.post(predict_url, files={"data": image}, timeout=400)
        if response.status_code == 200:
            print("Detection was successful!", response.json())
            return response.json()
        else:
            error_message = f"Backend returned status code {response.status_code}: {response.text}"
            print(error_message)
            return {"error": error_message}

    except Timeout:
        error_message = "The request to the backend timed out. Please try again later."
        print(error_message)
        return {"error": error_message}

    except RequestException as e:
        error_message = f"An error occurred while making the request: {str(e)}"
        print(error_message)
        return {"error": error_message}


def plot_confidence_histogram(data: torch.Tensor, bins: int = 20) -> plt.Figure:
    """
    Plot a histogram of the confidence scores with improved aesthetics and dark mode compatibility.
    :param data: The confidence scores
    :param bins: The number of bins for the histogram
    :param theme: 'dark' or 'light' theme
    :return: The histogram plot
    """
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="none")

    # Create the histogram
    n, bins, patches = ax.hist(data, bins=bins, edgecolor="white", alpha=0.7, linewidth=0.7, color="deepskyblue")

    # Set title and labels with appropriate colors
    title_color = "white"
    label_color = "white"

    ax.set_title("Confidence Scores Distribution", color=title_color, fontsize=16, pad=15)
    ax.set_xlabel("Confidence Score", color=label_color, fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency", color=label_color, fontsize=12, labelpad=10)

    # Customize tick parameters
    ax.tick_params(axis="x", colors=label_color, labelsize=10)
    ax.tick_params(axis="y", colors=label_color, labelsize=10)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_edgecolor(label_color)

    # Add gridlines for better readability
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.3)

    return fig


def main() -> None:
    """
    Main function of the Streamlit frontend.
    """
    st.set_page_config(page_title="Starfish Detection", layout="centered")

    # Connect to the backend service
    backend_url = "https://backend-638730968773.us-central1.run.app/"

    # Prompt the user to upload an image
    st.title("Starfish Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        st.image(image, caption="Uploaded Image", width=800, channels="BGR")

        with st.spinner("Detecting starfish..."):
            # Convert image to numpy array
            result = object_detection(image, backend=backend_url)

        if result is not None:
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Detection successful!")
                # Convert the contents of the result dictionary (lists) to tensors
                for key in result:
                    result[key] = torch.tensor(result[key])
                # Cnvert the image to a numpy array
                image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                processed_image = process_result(result, image)
                # Resize the image to the original size
                processed_image = cv2.resize(processed_image, (image.shape[1], image.shape[0]))

                # show the image and prediction
                st.image(processed_image, caption="Detected Starfish", width=800, channels="BGR")

                # Make a histogram of the scores
                fig = plot_confidence_histogram(result["scores"])
                st.pyplot(fig)

                # Show the predicted bounding boxes
                st.write("### Predicted Bounding Boxes:")
                st.write(result["boxes"])
        else:
            st.error("Failed to get prediction")


if __name__ == "__main__":
    main()
