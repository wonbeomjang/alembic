from typing import Any

import lightning
from lightning import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import functional as FM

from vision.configs import trainer as trainer_config
from vision.dataloader import get_dataloader
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

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)

        acc = FM.accuracy(
            preds,
            labels,
            self.config.task,
            num_classes=self.config.classification_model.num_classes,
        )
        metrics = {"val_acc": acc, "val_loss": loss}

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

    def configure_optimizers(self) -> optim.Optimizer:
        return get_optimizer(self.config.optimizer)(self.parameters())


class ClassificationTrainer(BasicTrainer):
    def __init__(
        self, config: trainer_config.Trainer, model: lightning.LightningModule
    ):
        self.model = model
        self.trainer = Trainer(max_epochs=config.epochs)
        self.train_loader = get_dataloader(config.train_data)

        if config.val_data is not None:
            self.val_loader = get_dataloader(config.val_data)
        else:
            self.val_loader = None

    def train(self):
        self.trainer.fit(self.model, self.train_loader)

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
