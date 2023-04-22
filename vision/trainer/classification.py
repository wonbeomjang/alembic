import time
from typing import Any, Union, Tuple, List, Dict

import lightning
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import functional as FM

from vision.configs import trainer as trainer_config
from vision.configs.optimizer import Optimizer
from vision.dataloader import get_dataloader
from vision.lr_scheduler import get_lr_scheduler
from vision.modeling import get_model
from vision.loss import get_loss
from vision.optimizer import get_optimizer
from vision.trainer import register_trainer
from vision.trainer._trainer import BasicTrainer


class Classification(lightning.LightningModule):
    def __init__(self, config: trainer_config.ClassificationTrainer):
        super().__init__()
        self.config = config

        self.model: nn.Module = get_model(config.classification_model)
        self.criterion: nn.Module = get_loss(config.loss)

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)

        acc = FM.accuracy(
            preds,
            labels,
            self.config.task,
            num_classes=self.config.classification_model.num_classes,
        )
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        cur_time = time.time_ns()
        preds = self(images)
        inference_time = (time.time_ns() - cur_time) / 1_000_000

        loss = self.criterion(preds, labels)

        acc = FM.accuracy(
            preds,
            labels,
            self.config.task,
            num_classes=self.config.classification_model.num_classes,
        )
        metrics = {
            "val_acc": acc,
            "val_loss": loss,
            "inference_time_ms": inference_time,
        }

        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        acc = FM.accuracy(
            preds,
            labels,
            self.config.task,
            num_classes=self.config.classification_model.num_classes,
        )
        loss = self.criterion(preds, labels)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, optim.lr_scheduler.LRScheduler):
            self.log_dict({"lr": lr_scheduler.get_last_lr()[0]})

    def configure_optimizers(
        self,
    ) -> Union[
        Tuple[List[Optimizer], List[Dict[str, Union[LRScheduler, str]]]], Optimizer
    ]:
        optimizer = get_optimizer(self.config.optimizer)(self.parameters())
        if self.config.lr_scheduler is not None:
            lr_scheduler = get_lr_scheduler(self.config.lr_scheduler)(
                optimizer, self.config.optimizer.lr
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        return optimizer


class ClassificationTrainer(BasicTrainer):
    def __init__(
        self, config: trainer_config.Trainer, model: lightning.LightningModule
    ):
        self.config = config
        logger = None
        checkpoint_callback = []
        if config.logger == "tensorboard":
            logger = TensorBoardLogger(save_dir=config.log_dir)

        if config.save_best_model:
            checkpoint_callback += [
                ModelCheckpoint(
                    dirpath=config.log_dir, filename="best.pt", monitor="val_acc"
                )
            ]

        self.model = model
        self.train_loader = get_dataloader(config.train_data)
        log_step = len(self.train_loader.dataset) // config.train_data.batch_size

        self.trainer = Trainer(
            max_epochs=config.epochs,
            logger=logger,
            callbacks=checkpoint_callback,
            log_every_n_steps=log_step,
        )

        if config.val_data is not None:
            self.val_loader = get_dataloader(config.val_data)
        else:
            self.val_loader = None

    def train(self):
        self.trainer.fit(self.model, self.train_loader, ckpt_path=self.config.ckpt)

    def train_and_eval(self):
        assert self.val_loader is not None
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def test(self, test_dataloader: DataLoader):
        self.trainer.test(self.model, test_dataloader)

    def predict(self, dataloader: DataLoader):
        self.trainer.predict(self.model, dataloader)


@register_trainer("classification")
def classification_trainer(config: trainer_config.Trainer):
    assert config.type == "classification"

    model = Classification(config.classification)
    trainer = ClassificationTrainer(config, model)

    return trainer
