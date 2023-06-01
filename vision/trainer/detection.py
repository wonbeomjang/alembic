import os
import time
from typing import Any, Union, Tuple, List, Dict, Optional

import lightning
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes, save_image

from vision.configs import task as trainer_config
from vision.configs.optimizer import Optimizer
from vision.dataloader import get_dataloader
from vision.lr_scheduler import get_lr_scheduler
from vision.modeling import get_model
from vision.loss import get_loss
from vision.modeling.yolo import YOLO
from vision.optimizer import get_optimizer
from vision.trainer import register_trainer
from vision.trainer._trainer import BasicTrainer
from vision.utils.coco import COCOEval
from vision.utils.common import STD, MEAN


class DetectionTask(lightning.LightningModule):
    def __init__(
        self, config: trainer_config.DetectionTask, coco_eval: Optional[COCOEval] = None
    ):
        super().__init__()
        self.config = config
        self.coco_eval = coco_eval

        self.model: YOLO = get_model(config.detection_model)
        self.criterion: nn.Module = get_loss(
            config.loss, box_coder=self.model.box_coder
        )

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        images, labels, _ = batch
        preds = self(images)

        loss: Dict[str, Tensor] = self.criterion(preds, labels, self.model.anchor)

        result_loss = sum(loss.values())
        loss = {"train/" + k: v for k, v in loss.items()}
        loss["train/loss"] = result_loss

        self.log_dict(loss)

        if result_loss != 0.0:
            return result_loss
        return None

    def validation_step(self, batch, batch_idx):
        images, labels, image_ids = batch

        cur_time = time.time_ns()
        preds = self(images)
        result = self.model.bbox_decoder(preds)

        if batch_idx == 0:
            single_image = images[0]
            single_bbox = result["bboxes"][0]
            single_score = result["scores"][0].detach().cpu().tolist()
            single_category_id = result["category_ids"][0].detach().cpu().tolist()

            label_txt = [
                f"{label}: {score:.2f}"
                for label, score in zip(single_category_id, single_score)
            ]
            single_image = (
                ((single_image * STD.to(self.device)) + MEAN.to(self.device)) * 255
            ).to(torch.uint8)
            single_image = draw_bounding_boxes(
                single_image, single_bbox, labels=label_txt, width=3
            )
            single_image = single_image.float() / 255

            save_image(single_image, "./image.png")

        inference_time = (time.time_ns() - cur_time) / 1_000_000

        loss = self.criterion(preds, labels, self.model.anchor)

        result_loss = sum(loss.values())
        loss = {"val/" + k: v for k, v in loss.items()}
        loss["val/loss"] = result_loss

        metrics = {
            "inference_time_ms": inference_time,
        }
        metrics.update(loss)

        self.log_dict(metrics)

        if self.coco_eval is not None:
            num_images = len(images)

            for i in range(num_images):
                image_id = torch.full_like(result["category_ids"][i], image_ids[i])
                self.coco_eval.update_dt(
                    image_id,
                    result["category_ids"][i],
                    result["bboxes"][i],
                    result["scores"][i],
                )

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    def on_train_epoch_end(self) -> None:
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, optim.lr_scheduler._LRScheduler):
            self.log_dict({"lr": lr_scheduler.get_last_lr()[0]})
        super().on_train_epoch_end()

    def on_validation_end(self) -> None:
        if self.coco_eval is not None:
            result = self.coco_eval.eval()
            self.log_dict(result)
        super().on_validation_end()

    def configure_optimizers(
        self,
    ) -> Union[
        Tuple[
            List[Optimizer],
            List[Dict[str, Union[optim.lr_scheduler._LRScheduler, str]]],
        ],
        Optimizer,
    ]:
        optimizer = get_optimizer(self.config.optimizer)(self.parameters())
        if self.config.lr_scheduler is not None:
            if self.config.lr_scheduler.total_steps is None:
                self.config.lr_scheduler.total_steps = self.config.total_steps

            lr_scheduler = get_lr_scheduler(self.config.lr_scheduler)(
                optimizer, self.config.optimizer.lr
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        return optimizer


class DetectionTrainer(BasicTrainer):
    def __init__(self, config: trainer_config.Trainer):
        self.config = config
        logger = None
        coco_eval = None
        checkpoint_callback = []
        if self.config.logger == "tensorboard":
            logger = TensorBoardLogger(save_dir=self.config.log_dir, name="")

        if self.config.save_best_model:
            checkpoint_callback += [
                ModelCheckpoint(
                    dirpath=self.config.log_dir,
                    filename="best",
                    monitor="val_acc",
                    save_last=True,
                )
            ]

        self.train_loader = get_dataloader(self.config.train_data)
        if self.config.val_data is not None:
            self.val_loader = get_dataloader(self.config.val_data)
            # coco_eval = COCOEval(self.val_loader.dataset.label_path)

        step_per_epochs = len(self.train_loader)

        if self.config.detection.total_steps is None:
            self.config.detection.total_steps = step_per_epochs * self.config.epochs

        if self.config.detection.detection_model.num_classes is None:
            self.config.detection.detection_model.num_classes = (
                self.train_loader.dataset.get_num_classes() + 1
            )

        self.model = DetectionTask(self.config.detection, coco_eval)

        last_path = os.path.join(config.log_dir, "last.ckpt")
        if os.path.exists(last_path):
            self.config.ckpt = last_path

        self.trainer = Trainer(
            max_epochs=self.config.epochs,
            logger=logger,
            callbacks=checkpoint_callback,
            log_every_n_steps=step_per_epochs,
        )

    def train(self):
        self.trainer.fit(self.model, self.train_loader, ckpt_path=self.config.ckpt)

    def eval(self):
        self.trainer.test(self.model, self.val_loader, ckpt_path=self.config.ckpt)

    def train_and_eval(self):
        assert self.val_loader is not None
        self.trainer.fit(
            self.model, self.train_loader, self.val_loader, ckpt_path=self.config.ckpt
        )

    def test(self, test_dataloader: DataLoader):
        self.trainer.test(self.model, test_dataloader)

    def predict(self, dataloader: DataLoader):
        self.trainer.predict(self.model, dataloader)


@register_trainer("detection")
def detection_trainer(config: trainer_config.Trainer):
    assert config.type == "detection"

    trainer = DetectionTrainer(config)

    return trainer
