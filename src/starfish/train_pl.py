from model import YOLO
from data import create_dataset

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


# Define the constants

DATA_PATH = "C:\\Users\\mjgoj\\Documents\\Data\\starfish_data"
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

MAX_EPOCHS = 50
BATCH_SIZE = 32

# 1. Create the dataset
dataset = create_dataset(DATA_PATH, subset=0.001)
# Split the dataset into training, validation, and test sets
train_size = int((1 - TEST_SPLIT - VAL_SPLIT) * len(dataset))
val_size = int(VAL_SPLIT * len(dataset))
test_size = len(dataset) - train_size - val_size
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# 2. Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Train the model

# Define model callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
# tf_logger = TensorBoardLogger("logs", name="yolo")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


model = YOLO()
trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'cpu', 
    max_epochs=MAX_EPOCHS, 
    callbacks=[early_stopping])
trainer.fit(model, train_loader, val_loader)