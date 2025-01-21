import pytest
import torch
from torch import Tensor
from starfish.model import FasterRCNNLightning

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = FasterRCNNLightning()
    x = torch.randn(1, 1, 640, 640)
    y = model(x)
    assert x.shape == (1, 1, 640, 640)
    assert y.shape == (batch_size, n_labels)
    
def test_error_on_wrong_shape():
    model = FasterRCNNLightning()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 640, 640]'):
        model(torch.randn(1,1,640,641))