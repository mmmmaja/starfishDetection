import pytorch_lightning as pl
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
import pandas as pd
import matplotlib.pyplot as plt


def compute_are_under_curve(precision, recall, verbose=True):
    if verbose:
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        # Set the limits
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.show()

    # at each recall level, we replace each precision value with the maximum precision value to the right of that recall level
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    # Compute the area under the curve
    area = 0
    for i in range(len(recall) - 1):
        area += (recall[i + 1] - recall[i]) * precision[i + 1]
    return area


def get_AP(scores, pred_boxes, gt_boxes, iou_threshold=0.5):
    sorted_indices = torch.argsort(scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]

    metrics = []
    used = torch.zeros(len(gt_boxes), dtype=torch.bool)
    true_positives, false_positives = 0, 0

    for i, pred_box in enumerate(pred_boxes):
        ious = torchvision.ops.box_iou(pred_box.unsqueeze(0), gt_boxes)
        iou_vals = ious[0]

        best_iou_val, best_gt_idx = torch.max(iou_vals, dim=0)
        if best_iou_val >= iou_threshold and not used[best_gt_idx]:
            true_positives += 1
            used[best_gt_idx] = True
        else:
            false_positives += 1

        precision = true_positives / (i + 1)
        recall = true_positives / len(gt_boxes)

        metrics.append([i, precision, recall])

    df = pd.DataFrame(metrics, columns=['rank', 'precision', 'recall'])

    # Compute AP as the area under the precision-recall curve
    return compute_are_under_curve(df['precision'], df['recall'])


def NMS(scores, boxes, iou_threshold=0.5):
    """
    TODO: fix this function
    Non-Maximum Suppression
    :param scores: Tensor of shape (N,) containing the confidence scores
    :param boxes: Tensor of shape (N, 4) containing the predicted boxes
    """

    # 1. Sort the predictions by confidence scores
    sorted_indices = torch.argsort(scores, descending=True)

    # 2. Create a list to store the indices of the predictions to keep
    keep_indices = []

    while len(sorted_indices) > 0:
        # Keep the prediction with the highest confidence score
        keep_indices.append(sorted_indices[0].item())

        # Calculate the IoU of the first prediction with all other predictions
        ious = torchvision.ops.box_iou(boxes[sorted_indices[0]].unsqueeze(0), boxes[sorted_indices])

        # Discard predictions with IoU greater than the threshold
        sorted_indices = sorted_indices[ious[0] <= iou_threshold]

    # Get the boxes and scores to keep
    keep_boxes = boxes[keep_indices]
    keep_scores = scores[keep_indices]
    return keep_scores, keep_boxes



class FasterRCNNLightning(pl.LightningModule):

    def __init__(self, num_classes, learning_rate=0.005, momentum=0.9, weight_decay=0.0005):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

        # Replace the classifier with a new one given number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = \
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        """
        Forward pass of the model
        :param images: Tensor of shape (N, C, H, W)
        """
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        """
        Training step
        :param batch: Tuple containing images and targets
        :param batch_idx: Index of the batch
        """
        images, targets = batch
        # Move targets to the same device as images
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass to obtain the losses
        loss_dict = self.model(images, targets)

        # Sum all losses (Categorical Cross-Entropy, Box, and Objectness)
        total_loss = sum(loss for loss in loss_dict.values())

        # Log individual losses and total loss
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)

        # FIXME: Not sure if it makes sense
        # Compute AP
        self.model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            predictions = self.model(images)  # Forward pass without targets
        self.model.train()

        # Extract scores and boxes
        scores = predictions[0]['scores']
        boxes = predictions[0]['boxes']
        targets_boxes = targets[0]['boxes']

        # Calculate the AP
        ap = get_AP(scores, boxes, targets_boxes)
        self.log('train_AP', ap, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        :param batch: Tuple containing images and targets
        :param batch_idx: Index of the batch
        """
        images, targets = batch
        # Move targets to the same device as images
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass without targets to get predictions
        predictions = self.model(images)

        # Calculate the AP
        scores = predictions[1]['scores']
        boxes = predictions[1]['boxes']
        targets_boxes = targets[1]['boxes']

        ap = get_AP(scores, boxes, targets_boxes)
        self.log('val_AP', ap, prog_bar=True)
        # TODO: add loss and log it


    def test_step(self, batch, batch_idx):
        """
        Test step
        :param batch: Tuple containing images and targets
        :param batch_idx: Index of the batch
        """

        images, targets = batch
        # Move targets to the same device as images
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass without targets to get predictions
        predictions = self.model(images)

        # Calculate the AP
        scores = predictions[1]['scores']
        boxes = predictions[1]['boxes']
        targets_boxes = targets[1]['boxes']

        ap = get_AP(scores, boxes, targets_boxes)
        self.log('test_AP', ap, prog_bar=True)


    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters()) # initializes the optimizer

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer) # initializes the scheduler

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
