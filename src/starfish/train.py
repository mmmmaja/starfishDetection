import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import numpy as np
import random
import os
from pathlib import Path

# Ensure reproducibility by setting seeds for random number generation
torch.manual_seed(409)
np.random.seed(409)
random.seed(409)

# Set CuBLAS workspace configuration for deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_path = str(Path(__file__).resolve().parent.parent.parent / "configs")

@hydra.main(config_path=config_path, config_name="main_config", version_base="1.2")
def train(cfg: DictConfig):
    # 1. Instantiate the data module
    data_module = instantiate(cfg.data)

    # 2. Instantiate the model
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)

    # 3. Instantiate the trainer
    trainer = instantiate(cfg.trainer, logger=logger)  # callbacks=[early_stopping], logger=True)

    # 4. Train the model
    trainer.fit(model, data_module)

    # 5. Test the model
    print("\nTesting the model...")
    trainer.test(model, data_module)

if __name__ == "__main__":
    train()
