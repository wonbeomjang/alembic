from abc import ABCMeta
from typing import Optional

import lightning
from lightning import Trainer
from torch.utils.data import DataLoader

from utils.logger import exception_console_logging, start_end_end_console_logging


class BasicTrainer(metaclass=ABCMeta):
    trainer: Optional[Trainer] = None
    model: Optional[lightning.LightningModule] = None
    train_loader: DataLoader = None
    val_loader: DataLoader = None
    config = None

    @exception_console_logging
    @start_end_end_console_logging(message="train")
    def train(self):
        self.trainer.fit(self.model, self.train_loader, ckpt_path=self.config.ckpt)

    @exception_console_logging
    @start_end_end_console_logging(message="evel")
    def eval(self):
        self.trainer.test(self.model, self.val_loader, ckpt_path=self.config.ckpt)

    @exception_console_logging
    @start_end_end_console_logging(message="train and eval")
    def train_and_eval(self):
        assert self.val_loader is not None
        self.trainer.fit(
            self.model, self.train_loader, self.val_loader, ckpt_path=self.config.ckpt
        )

    @exception_console_logging
    @start_end_end_console_logging(message="test")
    def test(self, test_dataloader: DataLoader):
        self.trainer.test(self.model, test_dataloader)

    @exception_console_logging
    @start_end_end_console_logging(message="predict")
    def predict(self, dataloader: DataLoader):
        self.trainer.predict(self.model, dataloader)
