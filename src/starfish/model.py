import pytorch_lightning as pl

"""
This file should contain the model definition
"""


def compute_metrics(y_hat, y):
    """
    TODO: Insert here the metrics we want to track
    Compute the metrics for the model.
    :param y_hat: The predicted values.
    :param y: The true values.
    """
    metrics_dict = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        'loss': None
    }
    return metrics_dict


class YOLO(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = None
        self.loss = None

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        return None

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: The input data.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        metrics = compute_metrics(y_hat, y)
        self.log_dict(metrics)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        metrics = compute_metrics(y_hat, y)
        self.log_dict(metrics)
        return metrics['loss']

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        metrics = compute_metrics(y_hat, y)
        self.log_dict(metrics)
        return metrics['loss']

