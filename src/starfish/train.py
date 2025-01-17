from model import FasterRCNNLightning
from data import create_dataset
from visualize import visualize_dataset
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import pytorch_lightning
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path, instantiate

@hydra.main(config_path="../../configs", config_name="main_config", version_base="1.2")
def train(cfg: DictConfig):
    # 1. Instantiate the data module
    data_module = instantiate(cfg.data)

    # 2. Instantiate the model
    model = instantiate(cfg.model)

    logger = instantiate(cfg.logger)

    # 3. Instantiate the trainer
    trainer = instantiate(cfg.trainer ,logger = logger)#callbacks=[early_stopping], logger=True)

    # 4. Train the model
    trainer.fit(model, data_module)

    # 5. Test the model
    print("\nTesting the model...")
    trainer.test(model, data_module)

    # 5. Load the best model
    # model = FasterRCNNLightning.load_from_checkpoint(checkpoint_path=trainer.checkpoint_callback.best_model_path, num_classes=2)
    # print("Model loaded successfully!")

if __name__ == "__main__":
    train()
