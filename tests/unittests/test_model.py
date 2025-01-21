import pytest
import torch
from torch import Tensor
from starfish.model import FasterRCNNLightning
from starfish.data import StarfishDataModule

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = FasterRCNNLightning().model
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    y = model(x)[0]
    boxes = y['boxes']
    assert boxes.shape[1] == 4, "Boxes shape is not correct"
    labels = y['labels']
    assert all(label in [0, 1] for label in labels), "Labels are not correct"
    scores = y['scores']
    assert all(score >= 0 and score <= 1 for score in scores), "Scores are not correct"

def test_train():
    fasterRCNN = FasterRCNNLightning()
    model = fasterRCNN.model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    datamodule = StarfishDataModule(data_from_bucket=False, subset=0.002)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    for batch in train_dataloader:
        images, targets = batch
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values()) 
        assert total_loss > 0, "Loss is not greater than 0"      
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
