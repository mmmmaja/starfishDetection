import torch
from pathlib import Path

parent_directory = Path(__file__).resolve().parents[2]


def train_yolov5(data_yaml_path, epochs=50, batch_size=16, img_size=640):
    """
    Train YOLOv5 model via Torch Hub.
    """

    # 1. Load a pretrained YOLOv5 model
    #    Options: 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', or the custom .pt
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # 2. Train the model
    results = model.train(
        data=data_yaml_path,                    # path to .yaml file
        epochs=epochs,                          # number of epochs
        batch_size=batch_size,                  # batch size
        imgsz=img_size,                         # size of the images (640 is typical)
        project=parent_directory / "reports",   # folder to save training results
        name="starfish_exp",                    # subfolder name
    )

    return model

if __name__ == '__main__':
    
    yaml_path = parent_directory / "data" / "processed" / "starfish_data.yaml"
    train_yolov5(yaml_path, epochs=1, batch_size=16, img_size=640)
