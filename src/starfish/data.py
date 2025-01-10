from pathlib import Path

import typer
from torch.utils.data import Dataset

"""
This script should include loading, cleaning, and splitting the data.

If the data needs to be pre-processed then running this file should process raw data in the data/raw 
folder and save the processed data in the data/processed folder.
"""


class StarfishDataset(Dataset):

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """

    def __getitem__(self, index: int):
        """
        Return a given sample from the dataset.
        """

    def preprocess(self, output_folder: Path) -> None:
        """
        Preprocess the raw data and save it to the output folder.
        """

    def plot_sample(self, index: int) -> None:
        """
        Plot a sample from the dataset.
        """


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """
    Preprocess the raw data and save it to the output folder.
    """
    print("Preprocessing data...")
    dataset = StarfishDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
