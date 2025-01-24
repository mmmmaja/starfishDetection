from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torchvision
import wandb
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FasterRCNNLightning(pl.LightningModule):
    """
    Lightning module for the Faster R-CNN model for starfish detection

    """

    def __init__(
        self,
        num_classes: int = 2,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.StepLR,
        compile: bool = False,
    ) -> None:
        super().__init__()
        """ Initializes the Faster R-CNN model lightning module
        :param num_classes: Number of classes
        :param optimizer: Optimizer
        :param scheduler: Learning rate scheduler
        :param compile: Whether to compile the model
        :param log_ap: Whether to log the average precision
        """
        self.save_hyperparameters()  # saves the hyperparameters to the checkpoint
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # loads the pretrained model

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # gets the number of input features

        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )  # replaces the pre-trained head with a new one

        self.val_map_metric = MeanAveragePrecision()
        self.val_iou_metric = IntersectionOverUnion()
        self.test_map_metric = MeanAveragePrecision()
        self.test_iou_metric = IntersectionOverUnion()

        torch.set_float32_matmul_precision("high")  # sets the precision to high

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Training step with Wandb image logging and bounding box overlays
        """
        images, targets = batch
        loss_dict = self.model(images, targets)  # forward pass with targets
        total_loss = sum(loss for loss in loss_dict.values())  # sum of all losses
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, prog_bar=True)  # logs all losses
        self.log("train_total_loss", total_loss, prog_bar=True)  # logs the total loss

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx) -> None:
        """
        Validation step
        :param batch: Tuple containing images and targets
        :param batch_idx: Index of the batch
        """
        self.model.eval()  # switches to evaluation mode

        images, targets = batch  # unpacks the batch
        predictions = self.model(images)  # forward pass without targets
        self.val_map_metric.update(predictions, targets)  # updates the mAP metric
        self.log_dict({f"val_{k}": v for k, v in self.val_map_metric.compute().items()}, prog_bar=True)  # logs the mAP
        self.val_iou_metric.update(predictions, targets)  # updates the IoU metric
        self.log_dict({f"val_{k}": v for k, v in self.val_iou_metric.compute().items()}, prog_bar=True)  # logs the IoU

        self.model.train()  # switches back to training mode
        loss_dict = self.model(images, targets)  # forward pass with targets
        total_loss = sum(loss for loss in loss_dict.values())  # sum of all losses
        self.log("val_total_loss", total_loss, prog_bar=True)  # logs the total loss

        return {
            "loss": total_loss,
            "images": images,
            "predictions": predictions,
            "targets": targets,
        }  # return dict with loss, images, predictions and targets to be used in callbacks

    def test_step(self, batch, batch_idx) -> None:
        """
        Test step
        :param batch: Tuple containing images and targets
        :param batch_idx: Index of the batch
        """

        images, targets = batch  # unpacks the batch
        predictions = self.model(images)  # forward pass without targets
        self.test_map_metric.update(predictions, targets)  # updates the mAP metric
        self.log_dict(
            {f"test_{k}": v for k, v in self.test_map_metric.compute().items()}, prog_bar=True
        )  # logs the mAP
        self.test_iou_metric.update(predictions, targets)  # updates the IoU metric
        self.log_dict(
            {f"test_{k}": v for k, v in self.test_iou_metric.compute().items()}, prog_bar=True
        )  # logs the IoU

        return {
            "images": images,
            "predictions": predictions,
            "targets": targets,
        }  # return dict with images, predictions and targets to be used in callbacks

    def setup(self, stage: str) -> None:
        """
        Setup the model
        :param stage: Stage of training
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler
        :return: Dictionary containing the optimizer and learning rate scheduler
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())  # initializes the optimizer
        print("learning rate", optimizer.param_groups[0]["lr"])
        print("momentum", optimizer.param_groups[0]["momentum"])

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)  # initializes the scheduler

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
