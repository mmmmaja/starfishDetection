from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity
from starfish.model import FasterRCNNLightning
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

image = Image.fromarray(
    np.uint8(np.random.rand(224, 224, 3) * 255)
)  # create a dummy image (e.g., 3-channel RGB image of size 224x224)
image_tensor = F.to_tensor(image)  # convert the image to a tensor
inputs = [image_tensor]
model = FasterRCNNLightning(num_classes=2)
model.eval()

with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("profiling")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
