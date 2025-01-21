from starfish.data import StarfishDataDataset
from starfish.data import StarfishDataModule


def test_data():
    """Test the MyDataset class."""
    dataset = Dataset("starfish-detection-data/data/raw")
    assert isinstance(dataset, Dataset)
    

    train, validation, test = create_dataset()
    assert image.shape == (1, 640, 640)
    assert len(train) == 1, "Dataset did not have the correct number of samples"
    assert len(validation) == 1, "Dataset did not have the correct number of samples"
    assert len(test) == 1, "Dataset did not have the correct number of samples"