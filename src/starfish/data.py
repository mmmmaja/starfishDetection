from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
import cv2
import torch
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from typing import Optional, Tuple
import pytorch_lightning as pl


def format_annotations(annotation_dict, image_width, image_height):
    """
    Format the annotations to [x_min, y_min, x_max, y_max] in absolute pixel coordinates for the fasterrcnn model.
    :param annotation_dict: The annotation dictionary with keys 'x', 'y', 'width', 'height'
    :param image_width: The width of the image
    :param image_height: The height of the image
    :return: The formatted annotations in the format [x_min, y_min, x_max, y_max, class_id]
    """

    # Original bounding box in pixel coords
    x_min = annotation_dict["x"]
    y_min = annotation_dict["y"]
    x_max = x_min + annotation_dict["width"]
    y_max = y_min + annotation_dict["height"]

    # Ensures that the coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)

    return [x_min, y_min, x_max, y_max, 1]  # class 1 denotes the starfish


def custom_collate_fn(batch):
    images = [sample[0] for sample in batch]  # List of tensors
    targets = [sample[1] for sample in batch]  # List of dicts
    return images, targets


class StarfishDataset(Dataset):
    def __init__(self, data_path: Path, transforms=None, subset=1.0) -> None:
        """
        Initialize the dataset.
        :param data_path: The path to the raw data
        :param transforms: The transformations to apply to the data from the albumentations library
        :param subset: The fraction of the data to load (the dataset is quite large)
        """
        self.data_path = data_path
        self.transforms = transforms

        # Load the data
        self.images, self.labels = self.load_files(subset)

    def load_files(self, subset: float):
        """
        Load images and their corresponding annotations.

        :param subset: Fraction of data to load
        :return: Tuple of lists (images, labels)
        """
        images, labels = [], []

        train_df = pd.read_csv(f"{self.data_path}/train.csv")
        # Remove the entries with empty annotations
        train_df = train_df[train_df.annotations != "[]"]
        # Take a subset of the data according to the subset parameter
        train_df = train_df.sample(frac=subset, random_state=42).reset_index(drop=True)
        print(f"Loading {len(train_df)} images.")

        # Load the images and annotations
        for _, row in train_df.iterrows():
            # Construct the image path
            image_path = f'{self.data_path}/train_images/video_{row["video_id"]}/{row["video_frame"]}.jpg'
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} does not exist. Skipping.")
                continue

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to read image {image_path}. Skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Read and process annotations
            raw_annotations = eval(row["annotations"])  # List of dicts
            if not raw_annotations:
                # If no annotations, skip this image
                continue

            formatted_annotations = [format_annotations(a, image.shape[1], image.shape[0]) for a in raw_annotations]

            images.append(image)
            labels.append(formatted_annotations)

        return images, labels

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int):
        """
        Return a given sample from the dataset.
        :param index: The index of the sample to return
        """
        image, labels = self.images[index], self.labels[index]
        # Change the image range from 0-255 to 0-1
        image = image / 255.0
        # Convert the image to a float tensor
        image = image.astype(np.float32)

        # Apply the transformations
        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=[label[:4] for label in labels], labels=[label[4] for label in labels]
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
        else:
            # If no transformations were provided, just return the image and labels
            boxes = [label[:4] for label in labels]
            labels = [label[4] for label in labels]

        target = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}

        return image, target

    def plot_sample(self, index: int, axs=None) -> None:
        """
        Plot a sample from the dataset.
        :param index: The index of the sample to plot
        """
        image, target = self[index]

        # Convert image tensor to numpy array
        image = image.permute(1, 2, 0).numpy()
        plot_show = False

        # Create a plot if axs is not provided
        if axs is None:
            plt.figure(figsize=(8, 8))
            axs = plt.gca()
            plot_show = True

        # Plot the image and the bounding boxes
        axs.imshow(image)
        for box in target["boxes"]:
            x, y, w, h = box
            rect = plt.Rectangle((x, y), w - x, h - y, fill=False, edgecolor="red", linewidth=2)
            axs.add_patch(rect)

        axs.axis("off")
        if plot_show:
            plt.show()


# def preprocess(raw_data_path: Path, output_folder: Path) -> None:
#     """
#     Preprocess the raw data and save it to the output folder.
#     :param raw_data_path: The path to the raw data
#     :param output_folder: The path to the output folder
#     """
#     print(f"Preprocessing data from {raw_data_path}...")
#     # From what I understand now we do need to do that
#     pass


def create_dataset(data_path, subset=1.0):
    """
    Create the dataset for training the model.
    :param data_path: The path to the raw data
    :param subset: Fraction of data to load
    """
    # Define the transformations to apply to the data
    # TODO: Normalize the images
    transform = A.Compose(
        [A.Resize(640, 640), A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.0, label_fields=["labels"]),
    )

    # Load the dataset
    return StarfishDataset(Path(data_path), subset=subset, transforms=transform)


class StarfishDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        subset: float = 0.002,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.subset = subset
        self.num_workers = num_workers

        # Define the transformations to apply to the data
        self.transforms = A.Compose(
            [A.Resize(640, 640), A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2), ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.0, label_fields=["labels"]),
        )

        # Initialize the datasets to None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def preprocess_data(self) -> None:
        """Process raw data and save it to the processed directory."""

    def setup(self, stage: str = None) -> None:
        """Load and prepare datasets."""

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = StarfishDataset(Path(self.data_path), transforms=self.transforms, subset=self.subset)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            shuffle=False,
        )


if __name__ == "__main__":
    # Get the main directory of the project
    parent_directory = Path(__file__).resolve().parents[2]
    data_path = parent_directory / "starfish-detection-data" / "data" / "raw"

    # For now the data does not require any preprocessing
    # Otherwise, uncomment the following line to preprocess the data
    # typer.run(preprocess)

    # Create the dataset and plot a sample
    dataset = create_dataset(data_path, subset=0.001)  # Adjust subset as needed
    for i in range(5):
        dataset.plot_sample(i)
