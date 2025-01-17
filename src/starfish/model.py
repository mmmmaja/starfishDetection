import pytorch_lightning as pl
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from typing import Dict, Any


def get_AP(scores, pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    TODO: test this function
    Since we have just one class AP is a sufficinet metric, we do need to average over classes
    :param scores: Tensor of shape (N,) containing the confidence scores
    :param pred_boxes: Tensor of shape (N, 4) containing the predicted boxes
    :param gt_boxes: Tensor of shape (M, 4) containing the ground truth boxes
    :param iou_threshold: IoU threshold to consider a prediction as correct
    """

    # 1. Create a list to store the confidence scores and IoU values for each prediction
    confidence_scores = []
    ious = []
    for i, pred_box in enumerate(pred_boxes):
        # Get the confidence score
        confidence_scores.append(scores[i])
        
        # Calculate the IoU with all ground truth boxes
        pred_box = pred_box.unsqueeze(0)
        iou_values = torchvision.ops.box_iou(pred_box, gt_boxes)
        ious.append(iou_values.max().item())

    # 2. Sort the confidence scores in descending order
    sorted_indices = torch.argsort(torch.tensor(confidence_scores), descending=True)

    # 3. Calculate the precision and recall at each confidence score threshold
    true_positives, false_positives = 0, 0
    precisions, recalls = [], []

    for i in sorted_indices:

        # Check if predicted box matches any ground truth box
        if ious[i] > iou_threshold:
            true_positives += 1
        else:
            false_positives += 1
        # Calculate precision and recall
        precisions.append(
            true_positives / (true_positives + false_positives)
        )
        recalls.append(
            true_positives / len(gt_boxes)
        )
    
    # Choose the max precision at each recall value and discard the rest
    max_precisions = []
    for recall in recalls:
        max_precisions.append(max([precision for i, precision in enumerate(precisions) if recalls[i] >= recall]))

    # Calculate the area under the precision-recall curve
    ap = 0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * max_precisions[i]
    return ap


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
    
    def __init__(self, 
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool, 
        log_ap: bool = False,
        learning_rate: float=0.005, 
        momentum: float=0.9, 
        weight_decay: float=0.0005):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Replace the classifier with a new one given number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = \
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        self.log_ap = log_ap

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
        # # Move targets to the same device as images
        # targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        # Forward pass to obtain the losses
        loss_dict = self.model(images, targets)     
    
        # Sum all losses (Categorical Cross-Entropy, Box, and Objectness)
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Log individual losses and total loss
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)

        if self.log_ap:
        # FIXME: Not sure if it makes sense
        # # Compute AP
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
        # # Move targets to the same device as images
        # targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        # # Forward pass without targets to get predictions
        predictions = self.model(images)
        
        # Calculate the AP 
        # FIXME: it somtimes throws an error if index isn't 0 but it should be 1?
        scores = predictions[0]['scores']
        boxes = predictions[0]['boxes']
        targets_boxes = targets[0]['boxes']

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
        
        # # Forward pass without targets to get predictions
        predictions = self.model(images)
        
        # # Calculate the AP
        scores = predictions[0]['scores']
        boxes = predictions[0]['boxes']
        targets_boxes = targets[0]['boxes']

        ap = get_AP(scores, boxes, targets_boxes)
        self.log('test_AP', ap, prog_bar=True)
 

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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
