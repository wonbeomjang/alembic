from abc import ABCMeta
from typing import Optional

try:
    from typing import Literal
except ModuleNotFoundError:
    from typing_extensions import Literal

import lightning
import torch
from lightning import Trainer

from vision.utils.load_weights import parse_state_dict
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

    def load_partial_state_dict(
        self, initial_weight_path: str, initial_weight_type: Literal["full", "backbone"]
    ):  # type: ignore
        org_state_dict = self.model.model.state_dict()
        trg_state_dict = torch.load(initial_weight_path)["state_dict"]
        trg_state_dict = parse_state_dict(trg_state_dict, initial_weight_type)  # type: ignore
        org_state_dict.update(trg_state_dict)
        self.model.model.load_state_dict(org_state_dict)
