import time
from typing import Any, Union, Tuple, List, Dict

import lightning
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn, optim, Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import functional as FM
from torchvision.transforms.functional import to_pil_image, to_tensor

from utils.cam.methods import GradCAMpp, CAM
from utils.cam.uitls import overlay_mask
from vision.configs import trainer as trainer_config
from vision.configs.optimizer import Optimizer
from vision.dataloader import get_dataloader
from vision.lr_scheduler import get_lr_scheduler
from vision.modeling import get_model
from vision.loss import get_loss
from vision.optimizer import get_optimizer
from vision.trainer import register_trainer
from vision.trainer._trainer import BasicTrainer


MEAN = torch.unsqueeze(torch.unsqueeze(torch.Tensor((0.485, 0.456, 0.406)), -1), -1)
STD = torch.unsqueeze(torch.unsqueeze(torch.Tensor((0.229, 0.224, 0.225)), -1), -1)


class ClassificationTask(lightning.LightningModule):
    def __init__(self, config: trainer_config.ClassificationTrainer):
        super().__init__()
        self.config = config

        self.model: nn.Module = get_model(config.classification_model)
        self.criterion: nn.Module = get_loss(config.loss)

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch

        if batch_idx == 0 and isinstance(self.logger, TensorBoardLogger):
            result = self.get_cam_image(image=images[0])
            self.logger.add_image("CAM_train", result, self.global_step)

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

        if batch_idx == 0 and isinstance(self.logger, TensorBoardLogger):
            result = self.get_cam_image(image=images[0])
            self.logger.experiment.add_image("CAM_val", result, self.global_step)

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

    def get_cam_image(self, image: Tensor) -> Tensor:
        numpy_image = (((image * STD.to(self.device)) + MEAN.to(self.device)) * 255).cpu().numpy().astype(np.uint8)
        numpy_image = np.transpose(numpy_image, [1, 2, 0])

        with CAM(self, target_layer="model.header.0", fc_layer="model.header.2") as cam_extractor:
            out = self(torch.unsqueeze(image, dim=0))
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            result = overlay_mask(to_pil_image(numpy_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            result = to_tensor(result)

        return result


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

    model = ClassificationTask(config.classification)
    trainer = ClassificationTrainer(config, model)

    return trainer
