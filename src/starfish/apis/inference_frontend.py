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
    

def plot_confidence_histogram(data: torch.Tensor, bins: int = 20) -> plt.Figure:
    """
    Plot a histogram of the confidence scores.
    :param data: The confidence scores
    :param bins: The number of bins for the histogram
    :return: The histogram plot
    """
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='none')

    colormap = plt.cm.get_cmap('viridis')
    # Create the histogram
    n, bins, patches = ax.hist(data, bins=bins, edgecolor='white', alpha=0.7, linewidth=0.7)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', colormap(c))
    
    # Set title and labels with light colors
    ax.set_title('Confidence Scores Distribution', color='white', fontsize=16, pad=15)
    ax.set_xlabel('Confidence Score', color='white', fontsize=12, labelpad=10)
    ax.set_ylabel('Frequency', color='white', fontsize=12, labelpad=10)

    # Customize tick parameters
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Add gridlines for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    return fig


def main() -> None:
    """
    Main function of the Streamlit frontend.
    """
    st.set_page_config(page_title="Starfish Detection", layout="centered")

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
                fig = plot_confidence_histogram(result['scores'])
                # Show the plot
                st.pyplot(fig)

                st.write("### Predicted Bounding Boxes:")
                st.write(result["boxes"])
        else:
            st.error("Failed to get prediction")


if __name__ == "__main__":
    main()