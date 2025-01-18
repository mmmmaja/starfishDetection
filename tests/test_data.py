from torch.utils.data import Dataset

from starfish.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("starfish-detection-data/data/raw")
    assert isinstance(dataset, Dataset)