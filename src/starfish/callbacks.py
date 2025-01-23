import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import wandb


class WandbImageLoggerCallback(Callback):
    def __init__(self, max_predictions=10, log_every_n_steps=100, num_images=4):
        """
        Callback for logging images with bounding boxes to WandB.

        :param confidence_threshold: Minimum confidence score for predictions to be logged.
        :param max_predictions: Maximum number of predictions to log per image.
        :param log_every_n_steps: Frequency of logging images (every n steps).
        """
        super().__init__()
        self.max_predictions = max_predictions
        self.log_every_n_steps = log_every_n_steps
        self.num_images = num_images

        self.train_map_metric = MeanAveragePrecision()
        self.train_iou_metric = IntersectionOverUnion()

    def log_images_with_boxes(self, pl_module, images, predictions, targets):
        """
        Log images with bounding boxes to WandB.
        """
        wandb_images = []

        for img, pred, tgt in zip(images[: self.num_images], predictions, targets):  # Log first num_images images
            img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert image to numpy (H, W, C)

            # Extract predictions
            pred_boxes = pred["boxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()

            # Get top N predictions by confidence
            top_indices = pred_scores.argsort()[::-1][: self.max_predictions]  # Sort scores descending and take top N
            pred_box_data = [
                {
                    "position": {
                        "minX": float(pred_boxes[i][0]) / img_np.shape[1],
                        "minY": float(pred_boxes[i][1]) / img_np.shape[0],
                        "maxX": float(pred_boxes[i][2]) / img_np.shape[1],
                        "maxY": float(pred_boxes[i][3]) / img_np.shape[0],
                    },
                    "class_id": int(pred_labels[i]),
                    "box_caption": f"Pred: {int(pred_labels[i])} ({pred_scores[i]:.2f})",
                    "scores": {"confidence": float(pred_scores[i])},
                }
                for i in top_indices
            ]

            # Process ground truth
            tgt_boxes = tgt["boxes"].cpu().numpy()
            tgt_labels = tgt["labels"].cpu().numpy()

            tgt_box_data = [
                {
                    "position": {
                        "minX": float(box[0]) / img_np.shape[1],
                        "minY": float(box[1]) / img_np.shape[0],
                        "maxX": float(box[2]) / img_np.shape[1],
                        "maxY": float(box[3]) / img_np.shape[0],
                    },
                    "class_id": int(label),
                    "box_caption": f"GT: {int(label)}",
                }
                for box, label in zip(tgt_boxes, tgt_labels)
            ]

            # Log image with class labels
            wandb_images.append(
                wandb.Image(
                    img_np,
                    boxes={
                        "predictions": {"box_data": pred_box_data, "class_labels": {0: "background", 1: "starfish"}},
                        "ground_truth": {"box_data": tgt_box_data, "class_labels": {0: "background", 1: "starfish"}},
                    },
                )
            )

        # Log to WandB
        if wandb_images:
            pl_module.logger.experiment.log({"training_images_with_boxes": wandb_images})  # step=pl_module.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Log mAP, IoU and images with bounding boxes to WandB at the end of each training batch.
        """
        if batch_idx % self.log_every_n_steps == 0:
            images, targets = batch

            pl_module.model.eval()  # switches to evaluation mode

            with torch.no_grad():  # disables gradient computation
                predictions = pl_module.model(images)  # forward pass without targets

            self.train_map_metric.update(predictions, targets)  # updates the mAP metric
            pl_module.log_dict(
                {f"train_{k}": v for k, v in self.train_map_metric.compute().items()}, prog_bar=True
            )  # logs the mAP
            self.train_iou_metric.update(predictions, targets)  # updates the IoU metric
            pl_module.log_dict(
                {f"train_{k}": v for k, v in self.train_iou_metric.compute().items()}, prog_bar=True
            )  # logs the IoU
            pl_module.model.train()  # switches back to training mode

            self.log_images_with_boxes(pl_module, images, predictions, targets)

        pl_module.model.train()  # switches back to training mode

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Log images with bounding boxes to WandB at the end of each validation batch.
        """

        if batch_idx % self.log_every_n_steps == 0:
            self.log_images_with_boxes(pl_module, outputs["images"], outputs["predictions"], outputs["targets"])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Log images with bounding boxes to WandB at the end of each test batch.
        """
        if batch_idx % self.log_every_n_steps == 0:
            self.log_images_with_boxes(pl_module, outputs["images"], outputs["predictions"], outputs["targets"])
