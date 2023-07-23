import os
from typing import Optional

from lightning import Trainer
from lightning.pytorch import callbacks
from lightning.pytorch import loggers
from torch.utils.data import DataLoader

from vision.configs.experiment import ExperimentConfig
from utils.logger import (
    exception_console_logging,
    start_end_end_console_logging,
    console_logger,
)
from vision.dataloader import get_dataloader
from vision.task.api import get_task
from vision.task.base import BaseTask


def _get_callbacks(config: ExperimentConfig):
    callback = [
        callbacks.LearningRateMonitor(logging_interval="epoch"),
        callbacks.RichProgressBar(),
        callbacks.Timer(),
    ]

    if config.save_best_model:
        callback += [
            callbacks.ModelCheckpoint(
                dirpath=config.log_dir,
                filename="best",
                monitor="val_acc",
                save_last=True,
                mode="max",
            )
        ]
    return callback


def _get_loggers(config: ExperimentConfig):
    logger = [loggers.CSVLogger(save_dir=config.log_dir, name=config.run_name)]

    if config.logger == "tensorboard":
        logger += [
            loggers.TensorBoardLogger(save_dir=config.log_dir, name=config.run_name)
        ]
    try:
        logger += [
            loggers.WandbLogger(save_dir=config.log_dir, project=config.run_name)
        ]
    except ModuleNotFoundError:
        pass

    return logger


class BaseTrainer:
    def __init__(self, config: ExperimentConfig):
        self.ckpt = None
        callback = _get_callbacks(config)
        logger = _get_loggers(config)

        self.train_loader: DataLoader = get_dataloader(config.train_data)
        self.val_loader: Optional[DataLoader] = (
            None if config.val_data is None else get_dataloader(config.val_data)
        )
        step_per_epochs = len(self.train_loader)

        if config.auto_num_classes:
            self.task: BaseTask = get_task(
                config.task,
                num_classes=self.train_loader.dataset.get_num_classes(),
                total_steps=step_per_epochs * config.epochs,
            )
        else:
            self.task: BaseTask = get_task(config.task)

        self.trainer = Trainer(
            max_epochs=config.epochs,
            logger=logger,
            callbacks=callback,
            log_every_n_steps=step_per_epochs,
        )

        # resume training
        last_path = os.path.join(config.log_dir, "last.ckpt")
        if config.load_last_weight and os.path.exists(last_path):
            console_logger.info(f"Load last weight from {last_path}")
            self.ckpt = last_path

    @exception_console_logging
    @start_end_end_console_logging(message="train")
    def train(self):
        self.trainer.fit(self.task, self.train_loader, ckpt_path=self.ckpt)

    @exception_console_logging
    @start_end_end_console_logging(message="evel")
    def eval(self):
        self.trainer.test(self.task, self.val_loader, ckpt_path=self.ckpt)

    @exception_console_logging
    @start_end_end_console_logging(message="train and eval")
    def train_and_eval(self):
        assert self.val_loader is not None
        self.trainer.fit(
            self.task, self.train_loader, self.val_loader, ckpt_path=self.ckpt
        )

    @exception_console_logging
    @start_end_end_console_logging(message="test")
    def test(self, test_dataloader: DataLoader):
        self.trainer.test(self.task, test_dataloader)

    @exception_console_logging
    @start_end_end_console_logging(message="predict")
    def predict(self, dataloader: DataLoader):
        self.trainer.predict(self.task, dataloader)
