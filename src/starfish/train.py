from model import FasterRCNNLightning
from data import create_dataset

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path


def custom_collate_fn(batch):
    images = [sample[0] for sample in batch]  # List of tensors
    targets = [sample[1] for sample in batch]  # List of dicts
    return images, targets


# Define the constants

parent_directory = Path(__file__).resolve().parents[2]
DATA_PATH = parent_directory / "data" / "raw"
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

MAX_EPOCHS = 1
BATCH_SIZE = 32

# 1. Create the dataset
dataset = create_dataset(DATA_PATH, subset=0.002)
# visualize_dataset(dataset, num_images=9)  # Comment out if you don't want to visualize the dataset

# Split the dataset into training, validation, and test sets
train_size = int((1 - TEST_SPLIT - VAL_SPLIT) * len(dataset))
val_size = int(VAL_SPLIT * len(dataset))
test_size = len(dataset) - train_size - val_size
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# 2. Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# 3. Train the model

# Define model callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
# tf_logger = TensorBoardLogger("logs", name="yolo")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


model = FasterRCNNLightning(num_classes=2)
trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=MAX_EPOCHS,
    default_root_dir=parent_directory,
    # callbacks=[early_stopping]
)
trainer.fit(model, train_loader, val_loader)

# 4. Test the model
print("\nTesting the model...")
trainer.test(model, test_loader)

# 5. Load the best model
model = FasterRCNNLightning.load_from_checkpoint(
    checkpoint_path=trainer.checkpoint_callback.best_model_path, num_classes=2
)
print("Model loaded successfully!")
