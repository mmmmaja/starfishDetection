import pytorch_lightning as pl
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch


def get_AP(scores, pred_boxes, gt_boxes):
    # Create a list to store the confidence scores and IoU values
    confidence_scores = []
    ious = []
    for i, pred_box in enumerate(pred_boxes):
        # Get the confidence score
        print(scores[i])
        confidence_scores.append(scores[i])
        
        # Calculate the IoU with all ground truth boxes
        pred_box = pred_box.unsqueeze(0)
        iou_values = torchvision.ops.box_iou(pred_box, gt_boxes)
        ious.append(iou_values.max().item())

    # Sort the confidence scores in descending order
    sorted_indices = torch.argsort(torch.tensor(confidence_scores), descending=True)

    true_positives, false_positives = 0, 0
    precisions, recalls = [], []
    for i in sorted_indices:
        print(f"Confidence: {confidence_scores[i]}, IoU: {ious[i]}")
        match = ious[i] > 0.5

        if match:
            true_positives += 1
        else:
            false_positives += 1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / len(gt_boxes)
        precisions.append(precision)
        recalls.append(recall)
    
    # Choose the max precision at each recall value
    max_precisions = []
    for recall in recalls:
        max_precisions.append(max([precision for p, r in zip(precisions, recalls) if r == recall]))

    # Calculate the area under the precision-recall curve
    ap = 0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * max_precisions[i]
    return ap



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
        
        # Forward pass
        loss_dict = self.model(images, targets)     
    
        # Sum all losses (Categorical Cross-Entropy, Box, and Objectness)
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Log individual losses and total loss
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)
        
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
        print(scores)
        print(boxes.shape, "boxes")
        print(targets_boxes.shape, "targets")

        ap = get_AP(scores, boxes, targets_boxes)
        self.log('val_AP', ap, prog_bar=True)
        # TODO: add loss and log it

        

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler
        """
        optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate,
                        momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
