import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity
from starfish.model import FasterRCNNLightning
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

# Create a dummy image (e.g., 3-channel RGB image of size 224x224)
image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

# Convert the image to a tensor
image_tensor = F.to_tensor(image)

# Create a dummy target dictionary
target = {
    "boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),  # Example bounding box
    "labels": torch.tensor([1], dtype=torch.int64),  # Example label
}

# The model expects a list of images and a list of targets
inputs = [image_tensor]
targets = [target]

model = FasterRCNNLightning(num_classes=2)
model.eval()

with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("profiling")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
