import pytorch_lightning as pl
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import sys

class FasterRCNNLightning(pl.LightningModule):
    
    def __init__(self, num_classes, learning_rate=0.005, momentum=0.9, weight_decay=0.0005):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier with a new one for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = \
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return losses
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        print(targets)
        sys.exit()
        loss_dict = self.model(images, targets)     
        # boxes, labels, scores = loss_dict[1]
        print(loss_dict[0])
        print(loss_dict[1])
        # 'boxes', 'labels', 'scores'
        # print(len(loss_dict), 'length of loss_dict', len(loss_dict))
        # for i in loss_dict:
        #     print(i.keys())
        sys.exit()
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Optionally, add metrics like mAP here
    
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate,
                        momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
