import torch  # Assuming it's a PyTorch model

# # Initialize the W&B API
# api = wandb.Api()
# # Replace with your artifact path
# artifact_path = "luciagordon-harvard-university/Starfish Detection/model-cqof85pt:v0"
# # Get the artifact
# artifact = api.artifact(artifact_path, type="model")
# # Download the artifact to a local directory
# artifact_dir = artifact.download()
# # Path to the model checkpoint file
# checkpoint_path = f"{artifact_dir}/model.ckpt"
from model import FasterRCNNLightning

import wandb

starfish_model = FasterRCNNLightning(num_classes=2)
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# state_dict = checkpoint['state_dict']

state_dict = torch.load("model.pth")
starfish_model.load_state_dict(state_dict)
starfish_model.eval()

for name, param in starfish_model.named_parameters():
    print(f"{name}: {param.shape}")

example_input = torch.rand(1, 3, 224, 224)
output = starfish_model(example_input)
print(output)
