from starfish.data import StarfishDataset
from starfish.data import StarfishDataModule
from tests import _PATH_DATA
import random
import os
import pytest

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_dataset():
    dataset = StarfishDataset(_PATH_DATA)
    assert len(dataset) == 4919, "Dataset did not have the correct number of samples"
    index = random.randint(0, 4918)
    image = dataset[index][0]
    assert image.shape == (720, 1280, 3), f"Image shape is not correct for index {index}"
    boxes = dataset[index][1]["boxes"]
    assert boxes.shape[1] == 4, f"Boxes shape is not correct for index {index}"
    labels = dataset[index][1]["labels"]
    assert all(label in [0, 1] for label in labels), f"Labels are not correct for index {index}"

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_datamodule():
    datamodule = StarfishDataModule(data_from_bucket=False)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    for dataloader in [train_dataloader, val_dataloader, test_dataloader]:
        image, target = next(iter(dataloader))
        assert image[0].shape == (3, 640, 640), "Image shape is not correct"
        boxes = target[0]['boxes']
        assert boxes.shape[1] == 4, f"Boxes shape is not correct"
        labels = target[0]["labels"]
        assert all(label in [0, 1] for label in labels), f"Labels are not correct"
