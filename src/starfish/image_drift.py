from pathlib import Path
import anyio
import nltk
import pandas as pd
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
from google.cloud import storage
from PIL import Image
import io
from fastapi import FastAPI
import json
import ast


BUCKET_NAME = "starfish-detection-data"

"""
Task: Deploy a drift detection API to the cloud (M27)
"""

def get_html(html_table, title):
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <!-- Bootstrap CSS -->
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



def extract_image_features(dataset):
    """
    Extract basic image features from a set of images (brightness, contrast, sharpness)
    and flatten them into scalar values.
    """

    feature_columns = [
        "Avg Brightness R", "Contrast R", "Sharpness R",
        "Avg Brightness G", "Contrast G", "Sharpness G",
        "Avg Brightness B", "Contrast B", "Sharpness B",
    ]

    features = []
    for i in range(len(dataset)):
        img = dataset[i]

        # Ensure img is a numpy array
        img = np.array(img)

        new_entry = []
        for channel_idx in range(3):
            channel = img[:, :, channel_idx]

            # Calculate average brightness
            avg_brightness = np.mean(channel)
            new_entry.append(avg_brightness)

            # Calculate contrast
            contrast = np.std(channel)
            new_entry.append(contrast)

            # Calculate sharpness
            sharpness = np.mean(np.abs(np.gradient(img)))
            new_entry.append(sharpness)

        features.append(new_entry)

    # Create DataFrame
    feature_df = pd.DataFrame(features, columns=feature_columns, index=range(len(dataset)))
    return feature_df


def parse_annotations(annotations):
    # Parse the annotations string into a list
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


def extract_target_features(targets_df):
    """
    Extract target features from the dataset.
    """
    target_columns = ["Avg BB size", "Number of BBs"]
    targets = []

    for idx, row in targets_df.iterrows():

        annotations = row.get("annotations", "")

        parsed_annotations = parse_annotations(annotations)
        print(f"{annotations} -> {parsed_annotations}")

        # If the list is empty, set the average size to 0
        if not parsed_annotations:
            targets.append([0, 0])

        else:
            print(parsed_annotations, "Annotations are not empty")
            total_size = 0

            for box in parsed_annotations:
                x_min, y_min = box["x"], box["y"]
                width, height = box["width"], box["height"]
                total_size += width * height
                print(total_size, "Total size")

            # Change total size to average size
            average_size = total_size / len(parsed_annotations)
            targets.append([average_size, len(parsed_annotations)])

    # Create DataFrame
    target_df = pd.DataFrame(targets, columns=target_columns, index=range(len(targets_df)))

    return target_df


def download_images(n: int = 5, prefix: str = "") -> None:
    """
    Download the N latest prediction files from the GCP bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    print(f"Acessing the bucket: {bucket}")
    blobs = bucket.list_blobs(prefix="")
    
    images = []
    idx = 0
    for blob in blobs:
        if blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_bytes = blob.download_as_bytes()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            images.append(img)
            idx += 1
            if idx >= n:
                break
    print(f"Download image data: ({len(images)})")
    return images


def download_targets(n: int = 5, prefix: str = "data/raw/train.csv") -> None:
    """
    Download the N latest prediction files from the GCP bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(prefix)
    
    if not blob.exists():
        raise FileNotFoundError(f"The file {prefix} does not exist in the bucket {BUCKET_NAME}.")
    else:
        print(f"Downloading {prefix} from the bucket {BUCKET_NAME}.")

    csv_bytes = blob.download_as_bytes()
    csv_str = csv_bytes.decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_str))
    # Get the first N rows
    df = df.head(n)
    return df

# Initialize FastAPI app
app = FastAPI()


@app.get("/report_images", response_class=HTMLResponse)
async def get_report_images(n: int = 5):
    """
    Generate and return the report.
    """
    data = download_images(n)
    
    # Get the statistics on the images
    image_features = extract_image_features(data)
    print(image_features)

     # Convert DataFrame to HTML
    html_table = image_features.to_html(classes='table table-striped', border=0)
    html_content = get_html(html_table, "Image Features Report")
    
    return HTMLResponse(content=html_content, status_code=200)

    # run_analysis(training_data, prediction_data)


    # async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
    #     html_content = f.read()

    # return HTMLResponse(content=html_content, status_code=200)


@app.get("/report_targets", response_class=HTMLResponse)
async def get_report_targets(n: int = 5):
    """
    Generate and return the report.
    """
    data = download_targets(n)
    
    # Get the statistics on the images
    target_features = extract_target_features(data)
    # Convert DataFrame to HTML
    html_table = target_features.to_html(classes='table table-striped', border=0)
    
    # Optional: Add some basic HTML structure
    html_content = get_html(html_table, "Target Features Report")
    
    return HTMLResponse(content=html_content, status_code=200)

    # run_analysis(trainidang_data, prediction_data)

    # async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
    #     html_content = f.read()

    # return HTMLResponse(content=html_content, status_code=200)



