import os
import random
from pathlib import Path

import albumentations as A
import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from loguru import logger as log
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

# Ensure reproducibility by setting seeds for random number generation
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
pl.seed_everything(0)


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # sets CuBLAS workspace configuration for deterministic behavior
torch.backends.cudnn.deterministic = True  # ensures that the CUDA backend produces deterministic results
torch.backends.cudnn.benchmark = False  # disables CuDNN benchmarking, which can introduce non-deterministic behavior

config_path = str(Path(__file__).resolve().parent.parent.parent / "configs")


@hydra.main(config_path=config_path, config_name="main_config", version_base="1.2")
def train(cfg: DictConfig):
    profiling = cfg.profiling
    log.info(f"Profiling = {profiling}")

    if profiling:
        prof = profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("profiling"))
        prof.start()

    # 1. Instantiate the data module
    data_module = instantiate(cfg.data)

    # 2. Instantiate the model
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)

    # Log the hyperparameters
    logger.log_hyperparams(cfg)

    # callbacks
    callbacks = [instantiate(cb) for _, cb in cfg.callbacks.items()]

    # 3. Instantiate the trainer
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)  # callbacks=[early_stopping], logger=True)

    # 4. Train the model
    log.info("\nTraining the model...")
    trainer.fit(model, data_module)

    if profiling:
        prof.stop()
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

    # 5. Test the model
    log.info("\nTesting the model...")
    trainer.test(model, data_module)


if __name__ == "__main__":
    train()
