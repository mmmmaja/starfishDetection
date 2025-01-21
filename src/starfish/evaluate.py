import torch
from model import FasterRCNNLightning

# starfish is a function that returns the training, validation and test sets
# from the data.py file
from data import starfish
from sklearn.metrics import classification_report
import typer
import numpy as np
import random
import os

# Ensure reproducibility by setting seeds for random number generation
torch.manual_seed(409)
np.random.seed(409)
random.seed(409)
# Set CuBLAS workspace configuration for deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

app.command()


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print(model_checkpoint)

    model = FasterRCNNLightning().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, _, test_set = starfish()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
        accuracy = correct / total
    print(f"Test accuracy: {accuracy}")

    report = classification_report(test_dataloader, y_pred)

    return accuracy, report


if __name__ == "__main__":
    app()
