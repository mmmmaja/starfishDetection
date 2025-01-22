import numpy as np
import pandas as pd
from evidently.metrics import DataDriftTable
from evidently.report import Report
from torchvision import transforms
from pathlib import Path
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import sys
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestMissingValues,
    TestNumericMetric,
    TestValueCount,
    TestNumericDistribution,
    TestDataDrift,
    TestDataQuality
)

"""
Task: Deploy a drift detection API to the cloud (M27)
"""


from data import StarfishDataset

image_transforms = A.Compose([
            A.Resize(640, 640),
        ], 
        bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0., label_fields=['labels'])
    )

parent_directory = Path(__file__).resolve().parents[2]
data_path = parent_directory / "data" / "raw"
print(f"Data path: {data_path}")
starfish_dataset = StarfishDataset(Path(data_path), subset=0.001, transforms=image_transforms)


def extract_image_features(dataset):
    """
    Extract basic image features from a set of images (brightness, contrast, sharpness)
    and flatten them into scalar values.
    """

    feature_columns = [
        "Average Brightness R", "Contrast R", "Sharpness R",
        "Average Brightness G", "Contrast G", "Sharpness G",
        "Average Brightness B", "Contrast B", "Sharpness B"
    ]
    

    features = []
    for i in range(len(dataset)):
        img, target = dataset[i]
        
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


def extract_target_features(dataset):
    """
    Extract target features from the dataset.
    """
    target_columns = ["Average BB size", "Number of BBs"]
    targets = []
    for i in range(len(dataset)):
        img, target = dataset[i]
        boxes = target["boxes"]
        total_size = 0

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            total_size += width * height

        # Change total size to average size
        average_size = total_size / len(boxes)
        # Convert to scalar
        average_size = average_size.item()
        targets.append([average_size, len(boxes)])
    
    # Create DataFrame
    target_df = pd.DataFrame(targets, columns=target_columns, index=range(len(dataset)))

    return target_df


# 1. Create a dataframe summarizing the image features
image_feature_df = extract_image_features(starfish_dataset)
print(image_feature_df.head())

# report = Report(metrics=[DataDriftTable()])
# report.run(reference_data=image_feature_df, current_data=image_feature_df)
# report.save_html("data_drift.html")

# 2. Create a dataframe summarizing the target features
target_feature_df = extract_target_features(starfish_dataset)
print(target_feature_df.head())


data_integrity_dataset_tests = TestSuite(
    tests=[
       # Make sure no images have zero bounding boxes
        TestValueCount(
            column_name="Number of BBs",
            values=[0],
            condition="==",
            target=0,
            assertion_type="less_than_or_equal"
        ),
    ]
)


# Create a report
data_integrity_dataset_tests.run(target_feature_df)

# Save the report
data_integrity_dataset_tests.save_report("data_integrity_dataset_tests.html")