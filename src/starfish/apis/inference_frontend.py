from google.cloud import run_v2
import streamlit as st
import os
import requests
from inference_backend import process_result
from requests.exceptions import Timeout, RequestException
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Constants for the Google Cloud project and region
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
        # print(service.name)
        if service.name.split("/")[-1] == "backend":
            print(f"Backend service found: {service.uri}")
            return service.uri
        
    name = os.environ.get("backend", None)
    return name


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
    

def plot_confidence_histogram(data: torch.Tensor, bins: int = 20, theme: str = 'dark') -> plt.Figure:
    """
    Plot a histogram of the confidence scores with improved aesthetics and dark mode compatibility.
    :param data: The confidence scores
    :param bins: The number of bins for the histogram
    :param theme: 'dark' or 'light' theme
    :return: The histogram plot
    """
    if theme == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='none')

    # Create the histogram
    n, bins, patches = ax.hist(data, bins=bins, edgecolor='white', alpha=0.7, linewidth=0.7)

    # Normalize bin counts for color mapping
    norm = plt.Normalize(min(n), max(n))
    colormap = plt.cm.viridis  # Suitable for dark backgrounds

    # Apply colors to each bin based on their normalized count
    for count, patch in zip(n, patches):
        color = colormap(norm(count))
        patch.set_facecolor(color)

    # Set title and labels with appropriate colors
    title_color = 'white' if theme == 'dark' else 'black'
    label_color = 'white' if theme == 'dark' else 'black'

    ax.set_title('Confidence Scores Distribution', color=title_color, fontsize=16, pad=15)
    ax.set_xlabel('Confidence Score', color=label_color, fontsize=12, labelpad=10)
    ax.set_ylabel('Frequency', color=label_color, fontsize=12, labelpad=10)

    # Customize tick parameters
    ax.tick_params(axis='x', colors=label_color, labelsize=10)
    ax.tick_params(axis='y', colors=label_color, labelsize=10)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_edgecolor(label_color)

    # Add gridlines for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    return fig


def main() -> None:
    """
    Main function of the Streamlit frontend.
    """
    st.set_page_config(page_title="Starfish Detection", layout="centered")

    # Sidebar for additional controls
    st.sidebar.header("Settings")

    # Theme selection (optional enhancement)
    theme = st.sidebar.radio("Select Theme", options=["Dark", "Light"], index=0)

     # Slider to adjust IoU threshold
    iou_threshold = st.sidebar.slider(
        "IoU Threshold for NMS",
        min_value=0.,
        max_value=1.,
        value=0.5,
        step=0.05,
        help="Adjust the Intersection over Union (IoU) threshold for Non-Maximum Suppression."
    )

    # Connect to the backend service
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        st.error(msg)
        return
    
    # Prompt the user to upload an image
    st.title("Starfish Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        st.image(image, caption="Uploaded Image", width=500, channels="BGR")

        with st.spinner("Detecting starfish..."):
            # Convert image to numpy array
            result = object_detection(image, backend=backend)

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
                st.image(processed_image, caption="Detected Starfish", width=500, channels="BGR")

                # Make a histogram of the scores                
                # Create the plot
                fig = plot_confidence_histogram(result['scores'], theme=theme.lower())
                # Show the plot
                st.pyplot(fig)

                st.write("### Predicted Bounding Boxes:")
                st.write(result["boxes"])
        else:
            st.error("Failed to get prediction")


if __name__ == "__main__":
    main()