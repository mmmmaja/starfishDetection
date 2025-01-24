import ast
import json
import os

import anyio
import cv2
import numpy as np
import pandas as pd
from evidently.metrics import ColumnDriftMetric, DataDriftTable
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Where the training data is stored
REFERENCE_BUCKET_URL = "/gcs/starfish-detection-data"
# Where the data is uploaded to when the inference API is called
CURRENT_BUCKET_URL = "/gcs/inference_api_data"

"""
Task: Deploy a drift detection API to the cloud (M27)
"""


def get_html(html_table: str, title: str) -> str:
    """
    Generate an HTML report with a title and a table
    :param html_table: The HTML table to include in the report
    :param title: The title of the report
    """
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <link
                rel="stylesheet"
                href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
                integrity="sha384-JcKb8q3iqJ61gNVX38n5IYjPjzq3jVV0T1J5i5x6d11s5jzVlae6q9wl8LCjhT1X"
                crossorigin="anonymous">
            <link
                rel="stylesheet"
                href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
                integrity="sha512-iBBXm8fW90+nuLcSKVBQOUCmg2nQ93Lj6V1QGNd/axh4K0bDjO/baMQVFcE6QeJ3Jxk4a+X9JfSZVv2y+I3BMQ=="
                crossorigin="anonymous" />
            <style>
                body {{
                    margin: 20px;
                    background-color: #f8f9fa;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}
                h1 {{
                    margin-bottom: 30px;
                    color: #343a40;
                    text-align: center;
                }}
                table {{
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th {{
                    background-color: #343a40;
                    color: white;
                    text-align: center;
                }}
                td {{
                    text-align: center;
                }}
                .table-container {{
                    overflow-x: auto;
                }}
                /* Hover effect for table rows */
                tbody tr:hover {{
                    background-color: #f1f1f1;
                }}
                .container-custom {{
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                @media (max-width: 768px) {{
                    h1 {{
                        font-size: 1.5rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container-custom">
                <h1>{title}</h1>
                <div class="table-container">
                    {html_table}
                </div>
            </div>
        </body>
    </html>
    """


def extract_color_histograms(img: np.ndarray, bins: int = 32) -> list:
    """
    Extract color histograms for each channel.
    :param img: The input image as a NumPy array.
    :param bins: Number of bins for the histogram.
    :return: A list containing histograms for each channel.
    """
    hist_features = []
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    return hist_features


def extract_image_features(dataset: list) -> pd.DataFrame:
    """
    Extract features from the image dataset.
    :param dataset: The list of images as PIL Image objects.
    :return: A DataFrame containing the extracted features.
    """

    feature_list = []
    for img in dataset:
        img_np = np.array(img)

        # Extract basic features
        avg_brightness_r = np.mean(img_np[:, :, 0])
        contrast_r = np.std(img_np[:, :, 0])
        sharpness_r = np.mean(np.abs(np.gradient(img_np[:, :, 0])))

        avg_brightness_g = np.mean(img_np[:, :, 1])
        contrast_g = np.std(img_np[:, :, 1])
        sharpness_g = np.mean(np.abs(np.gradient(img_np[:, :, 1])))

        avg_brightness_b = np.mean(img_np[:, :, 2])
        contrast_b = np.std(img_np[:, :, 2])
        sharpness_b = np.mean(np.abs(np.gradient(img_np[:, :, 2])))

        # Color Histograms
        color_hist = extract_color_histograms(img_np)

        # Combine all features
        combined_features = [
            avg_brightness_r,
            contrast_r,
            sharpness_r,
            avg_brightness_g,
            contrast_g,
            sharpness_g,
            avg_brightness_b,
            contrast_b,
            sharpness_b,
        ] + color_hist

        feature_list.append(combined_features)

    # Define feature column names
    feature_columns = [
        "Avg Brightness R",
        "Contrast R",
        "Sharpness R",
        "Avg Brightness G",
        "Contrast G",
        "Sharpness G",
        "Avg Brightness B",
        "Contrast B",
        "Sharpness B",
    ]
    # Add color histogram columns
    color_hist_bins = 32
    for channel in ["R", "G", "B"]:
        for bin_idx in range(color_hist_bins):
            feature_columns.append(f"ColorHist_{channel}_{bin_idx}")

    # Create DataFrame
    feature_df = pd.DataFrame(feature_list, columns=feature_columns, index=range(len(dataset)))
    return feature_df


def parse_annotations(annotations: str) -> list:
    """
    Parse the annotations string into a list of dictionaries.
    :param annotations: The annotations string
    :return: A list of dictionaries containing the parsed annotations
    """
    try:
        parsed_annotations = ast.literal_eval(annotations)
    except json.JSONDecodeError:
        # Handle cases where annotations are not valid JSON
        parsed_annotations = []
    except Exception as e:
        # Handle any other exceptions
        parsed_annotations = []

    # Check if parsed_annotations is a list
    if not isinstance(parsed_annotations, list):
        parsed_annotations = []

    return parsed_annotations


def extract_target_features(targets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract target features from the dataset.
    :param targets_df: The DataFrame containing the target data
    :return: A DataFrame containing the extracted features
    """
    target_columns = ["Avg BB size", "Number of BBs"]
    targets = []

    for _, row in targets_df.iterrows():
        annotations = row.get("annotations", "")
        parsed_annotations = parse_annotations(annotations)

        # If the list is empty, set the average size to 0
        if not parsed_annotations:
            targets.append([0, 0])

        else:
            # Else compute the average size of the bounding boxes
            total_size = 0

            for box in parsed_annotations:
                width, height = box["width"], box["height"]
                total_size += width * height
                print(total_size, "Total size")

            # Change total size to average size
            average_size = total_size / len(parsed_annotations)
            targets.append([average_size, len(parsed_annotations)])

    # Create DataFrame
    target_df = pd.DataFrame(targets, columns=target_columns, index=range(len(targets_df)))

    return target_df


def download_images(bucket_name: str, n: int = 5, prefix: str = "data/raw/train_images") -> list:
    """
    Download the N latest prediction files from the GCP bucket (training data).
    :param n: The number of images to download
    :param prefix: The prefix of the files to download
    :return: A list of PIL Image objects
    """
    data_path = f"{bucket_name}/{prefix}"

    images, idx = [], 0
    for folder in os.listdir(data_path):
        if prefix == 'uploaded_images':
            file = os.path.join(data_path, folder)

            if file.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(data_path, folder, file)
                # Load the image
                img = cv2.imread(img_path)
                images.append(img)
        else:
            for file in os.listdir(os.path.join(data_path, folder)):
                if file.endswith((".jpg", ".jpeg", ".png")):
                    idx += 1
                    img_path = os.path.join(data_path, folder, file)
                    # Load the image
                    img = cv2.imread(img_path)
                    images.append(img)

                    if idx >= n:
                        break
    print(f"Download image data: ({len(images)})")
    return images


def download_targets(bucket_name: str, n: int = 5, prefix: str = "data/raw/train.csv") -> None:
    """
    Download the N latest prediction files from the GCP bucket.
    :param bucket_name: The name of the GCP bucket
    :param n: The number of images to download
    :param prefix: The prefix of the files to download
    :return: A DataFrame containing the target data
    """
    data_path = f"{bucket_name}/{prefix}"
    try:
        # Read the csv file
        df = pd.read_csv(data_path)
        print(f"Download target data: ({len(df)})")
        df = df.head(n)
        return df
    except Exception as e:
        print(f"Error downloading target data: {e}")
        return None


# Initialize FastAPI app
app = FastAPI()


@app.get("/report_images", response_class=HTMLResponse)
async def get_report_images(n: int = 5) -> HTMLResponse:
    """
    Generate and return the report on the image data (for the training data).
    :param n: The number of images to include in the report
    :return: The HTML response containing the report
    """
    data = download_images(REFERENCE_BUCKET_URL, n)

    # Get the statistics on the images
    image_features = extract_image_features(data)

    # Convert DataFrame to HTML
    html_table = image_features.to_html(classes="table table-striped", border=0)
    html_content = get_html(html_table, "Image Features Report")

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/report_targets", response_class=HTMLResponse)
async def get_report_targets(n: int = 5) -> HTMLResponse:
    """
    Generate and return the report on the target data (for the training data).
    :param n: The number of targets to include in the report
    :return: The HTML response containing the report
    """
    data = download_targets(REFERENCE_BUCKET_URL, n)

    # Get the statistics on the images
    target_features = extract_target_features(data)

    # Convert DataFrame to HTML
    html_table = target_features.to_html(classes="table table-striped", border=0)

    # Optional: Add some basic HTML structure
    html_content = get_html(html_table, "Target Features Report")

    return HTMLResponse(content=html_content, status_code=200)


def remove_zero_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features with zero variance.
    :param df: Input DataFrame.
    :return: DataFrame with zero variance features removed.
    """
    return df.loc[:, df.var() > 0]


@app.get("/drift_report", response_class=HTMLResponse)
async def get_drift_report(n: int = 5) -> HTMLResponse:
    """
    Generate and return the drift report between the reference and current data.
    :param n: The number of images to include in the report
    :return: The HTML response containing the drift report
    """

    # Download the images from the GCP buckets
    reference_images = download_images(REFERENCE_BUCKET_URL, n)
    current_images = download_images(CURRENT_BUCKET_URL, n, prefix='uploaded_images')

    # Extract features from the datasets
    reference_data = extract_image_features(reference_images)
    current_data = extract_image_features(current_images)

    # Remove zero variance features
    reference_data = remove_zero_variance_features(reference_data)
    current_data = remove_zero_variance_features(current_data)

    if len(current_images) == 0 or len(reference_images) == 0:
        return HTMLResponse(content="No images found in the buckets.", status_code=404)

    # Generate the drift report
    report = Report(
        metrics=[
            DataDriftTable(),
            ColumnDriftMetric(column_name="Avg Brightness R"),
            ColumnDriftMetric(column_name="Contrast R"),
            ColumnDriftMetric(column_name="Sharpness R"),
        ]
    )
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("monitoring.html")

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)


# Define the root endpoint
@app.get("/")
async def root() -> dict:
    """
    Root endpoint.
    """
    return {"message": "Hello from the data drift module!"}
