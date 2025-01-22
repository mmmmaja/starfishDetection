from typing import List

import numpy as np
import onnxruntime as rt
import torch
import typer
from model import FasterRCNNLightning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

app.command()


@app.command()
def export_to_onnx(model_checkpoint: str, onnx_file_path: str):
    """Export a model to ONNX"""
    print(model_checkpoint)

    model = FasterRCNNLightning().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    dummy_input = torch.randn(1, 3, 800, 800).to(DEVICE)

    onnx_model = torch.onnx.dynamo_export(
        model=model,
        model_args=(dummy_input,),
        export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
    )

    onnx_model.save(onnx_file_path)

    print(f"Model successfully exported to {onnx_file_path}")


@app.command()
def run_onnx_model(onnx_file_path: str) -> List[np.ndarray]:
    """Run ONNX model"""
    ort_session = rt.InferenceSession(onnx_file_path)
    input_names = [i.name for i in ort_session.get_inputs()]
    output_names = [i.name for i in ort_session.get_outputs()]
    batch = {input_names[0]: np.random.randn(1, 3, 224, 224).astype(np.float32)}
    out = ort_session.run(output_names, batch)

    return out


if __name__ == "__main__":
    app()
