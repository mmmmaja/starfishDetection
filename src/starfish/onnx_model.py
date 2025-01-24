from typing import List

import numpy as np
import onnxruntime as rt
import torch
import typer
from model import FasterRCNNLightning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = typer.Typer()
app.command()


@app.command()
def export_to_onnx(onnx_file_path: str) -> None:
    """Export a model to ONNX"""
    model_path = "https://storage.googleapis.com/starfish-model/model.ckpt"
    model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=model_path, num_classes=2)
    model = FasterRCNNLightning().to(DEVICE)
    model.eval()

    dummy_input = torch.randn(1, 3, 800, 800).to(DEVICE)

    torch.onnx.export(
        model.model,
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
