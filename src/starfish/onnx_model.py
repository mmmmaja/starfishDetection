from typing import List

import numpy as np
import onnxruntime as rt
import torch
import typer
from model import FasterRCNNLightning

import wandb

DEVICE = torch.device("cpu")
app = typer.Typer()
app.command()


@app.command()
def export_to_onnx(artifact_name: str, onnx_file_path: str):
    """Export a model to ONNX"""
    wandb.init(project="starfishDetection-src_starfish")
    artifact = wandb.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download()
    checkpoint_path = f"{artifact_dir}/model.ckpt"

    model = FasterRCNNLightning().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 800, 800).to(DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model successfully exported to {onnx_file_path}")


@app.command()
def run_onnx_model(onnx_file_path: str) -> List[np.ndarray]:
    """Run ONNX model"""
    ort_session = rt.InferenceSession(onnx_file_path)
    input_names = [i.name for i in ort_session.get_inputs()]
    output_names = [i.name for i in ort_session.get_outputs()]
    batch = {input_names[0]: np.random.randn(1, 3, 800, 800).astype(np.float32)}
    out = ort_session.run(output_names, batch)

    return out


if __name__ == "__main__":
    app()
