import random
import time
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset

from solver import Solver


@hydra.main(version_base=None, config_path='../conf', config_name='unsupervised_phoneme_seg')
def main(config: DictConfig):
    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    # cudnn.benchmark = True

    # set model path
    config.model_save_path = join(config.model_save_path, "unsupervised")
    logger.info(f"saving models in:: {config.model_save_path}")

    # set callbacks
    early_stopping = EarlyStopping(monitor=config.early_stop_metric, patience=5, verbose=True, mode=config.early_stop_mode)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_save_path,
        filename="pho_seg_{epoch:02d}_{val_loss:.2f}",
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor=config.early_stop_metric,
        mode=config.early_stop_mode,
    )

    # set logger
    # wandb_logger = WandbLogger(project=config.project, name=config.name)
    csv_logger = CSVLogger(save_dir=config.model_save_path, name="log")
    logger.add(join(config.model_save_path, "log.csv"), rotation="10 MB", level="INFO")
    logger.info(f'saving log: log.csv')

    # set solver
    solver = Solver(config)

    # set trainer
    trainer = Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.epochs,
        callbacks=[early_stopping, checkpoint_callback],
        logger=csv_logger,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        val_check_interval=config.val_check_interval,
        fast_dev_run=config.dev_run,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.accumulate_grad_batches
    )


    if not config.test:
        # train
        trainer.fit(solver)

    if config.ckpt is not None:
        ckpt = config.ckpt
    else:
        ckpt = solver.get_ckpt_path()

    # test
    print(f"running test on ckpt: {ckpt}")
    solver = Solver.load_from_checkpoint(ckpt)
    trainer.test(solver)


if __name__ == "__main__":
    main()
