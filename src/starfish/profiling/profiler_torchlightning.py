import torch
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import PyTorchProfiler
from starfish.model import FasterRCNNLightning
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os

# Create a dummy image (e.g., 3-channel RGB image of size 224x224)
image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

# Convert the image to a tensor
image_tensor = F.to_tensor(image)

# Create a dummy target dictionary
target = {
    "boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),  # Example bounding box
    "labels": torch.tensor([1], dtype=torch.int64)  # Example label
}

# The model expects a list of images and a list of targets
inputs = [image_tensor]
targets = [target]

model = FasterRCNNLightning(num_classes=2)
model.eval()

log_dir = "./profilers/log/resnet18"
os.makedirs(log_dir, exist_ok=True)

# Create a PyTorch Profiler
profiler = PyTorchProfiler(
    dirpath=log_dir,
    filename="profiler_logs",
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
)

# Create a Trainer with the profiler
trainer = Trainer(profiler=profiler, max_epochs=1)

# Train the model
trainer.fit(model, train_dataloaders=[inputs], val_dataloaders=[targets])

print(profiler.summary())