import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import sys


"""
This file should contain the model definition
"""


class YOLO(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # Load the pre-trained YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # TODO change this to hparams
        self.learning_rate = 1e-3
        self.weight_decay = 0.0005

    def configure_optimizers(self):
        """
        Configure the optimizer and the scheduler.
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: The input data.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        results = self.model(x)
        print(results)
        sys.exit()
        loss = results.loss
        predictions = results.pred

        metrics = {
            "loss": loss.item(),
            "MAP": MeanAveragePrecision(num_classes=1)(predictions, y)
        }
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        results = self.model(x)
        print(results.pred)
        sys.exit()
        
        loss = results.loss
        predictions = results.pred

        metrics = {
            "val_loss": loss.item(),
            "val_MAP": MeanAveragePrecision(num_classes=1)(predictions, y)
        }
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        results = self.model(x)

        loss = results.loss
        predictions = results.pred

        metrics = {
            "test_loss": loss.item(),
            "test_MAP": MeanAveragePrecision(num_classes=1)(predictions, y)
        }
        self.log_dict(metrics)
        return loss
        
        

