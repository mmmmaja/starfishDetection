import pytorch_lightning as pl
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torchmetrics.classification import BinaryAveragePrecision
import pandas as pd
import matplotlib.pyplot as plt


def compute_are_under_curve(precision, recall, verbose=False):
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
    """
    Compute Average Precision (AP) for a single class.
    
    :param scores:      Tensor of shape (N,) containing the confidence scores
    :param pred_boxes:  Tensor of shape (N, 4) containing the predicted boxes
    :param gt_boxes:    Tensor of shape (M, 4) containing the ground truth boxes
    :param iou_threshold: IoU threshold to consider a prediction as correct
    
    :return: AP (float)
    """

    # Sort predictions by confidence desc
    sorted_indices = torch.argsort(scores, descending=True)
    # Sort predictions by confidence in descending order
    pred_boxes = pred_boxes[sorted_indices]

    # Create the dataframe to track the metrics
    df = pd.DataFrame(columns=['rank', 'correct', 'precision', 'recall'])
    
    # For each prediction determine if it matches an unused GT
    used = torch.zeros(len(gt_boxes), dtype=torch.bool)  # track which GT boxes are used
    
    true_positives, false_positives = [], []
    
    for i, pred_box in enumerate(pred_boxes):
        # Compute IoU with all GT boxes
        ious = torchvision.ops.box_iou(pred_box.unsqueeze(0), gt_boxes)  # shape (1, M)
        iou_vals = ious[0]
        
        # Find the best GT match (highest IoU) above the threshold
        best_iou_val, best_gt_idx = torch.max(iou_vals, dim=0)
        
        if best_iou_val >= iou_threshold and not used[best_gt_idx]:
            # This is a true positive and we mark that GT box as used
            true_positives.append(1)
            false_positives.append(0)
            used[best_gt_idx] = True
            correct = True
        else:
            # Otherwise, it's a false positive
            true_positives.append(0)
            false_positives.append(1)
            correct = False

        precision = torch.sum(torch.tensor(true_positives, dtype=torch.float)) / (i + 1)
        recall = torch.sum(torch.tensor(true_positives, dtype=torch.float)) / len(gt_boxes)
        df.loc[i] = [i, correct, precision, recall]
    
    # Plot the precision-recall curve
    precision = df['precision'].to_numpy()
    recall = df['recall'].to_numpy()

    return compute_are_under_curve(precision, recall)


def NMS(scores, boxes, iou_threshold=0.5):
    """
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
        optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate,
                        momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]