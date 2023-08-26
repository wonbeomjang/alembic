import time
from typing import Any, Union, Tuple, List, Dict, Optional

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, optim, Tensor
from torchvision.utils import draw_bounding_boxes

from vision.configs import task as task_config
from vision.configs.optimizer import Optimizer
from vision.configs.task import Task
from vision.lr_scheduler import get_lr_scheduler
from vision.modeling import get_model
from vision.loss import get_loss
from vision.optimizer import get_optimizer
from vision.task.api import register_task
from vision.utils.coco import COCOEval
from vision.utils.common import STD, MEAN
from utils.logger import console_logger


class DetectionTask(lightning.LightningModule):
    def __init__(
        self,
        config: task_config.DetectionTask,
        coco_eval: Optional[COCOEval] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.coco_eval = coco_eval

        self.model = get_model(config.model, **kwargs)
        self.criterion: nn.Module = get_loss(
            config.loss, box_coder=self.model.box_coder
        )

        self.initialize()

    def initialize(self):
        if self.config.initial_weight_path is not None:
            console_logger.info(
                f"Load {self.config.initial_weight_type} weight from {self.config.initial_weight_path}"
            )
            self.load_partial_state_dict(
                self.config.initial_weight_path, self.config.initial_weight_type
            )  # type: ignore

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

        if batch_idx == 0:
            result = self.model.bbox_decoder(preds)
            dt_image = self.draw_bbox(
                images[0],
                result["bboxes"][0],
                result["scores"][0],
                result["category_ids"][0],
            )
            gt_image = self.draw_bbox(
                images[0],
                labels[0]["boxes"],
                torch.ones(labels[0]["labels"].shape),
                labels[0]["labels"],
            )
            self.logger.experiment.add_image(
                "train/dt_image",
                dt_image,
                self.global_step,
            )
            self.logger.experiment.add_image(
                "train/gt_image",
                gt_image,
                self.global_step,
            )

        if result_loss != 0.0:
            return result_loss
        return None

    def validation_step(self, batch, batch_idx):
        images, labels, image_ids = batch

        cur_time = time.time_ns()
        preds = self(images)

        inference_time = (time.time_ns() - cur_time) / 1_000_000

        loss = self.criterion(preds, labels, self.model.anchor)

        result_loss = sum(loss.values())
        loss = {"val/" + k: v for k, v in loss.items()}
        loss["val/loss"] = result_loss

        metrics = {
            "inference_time_ms": inference_time,
        }
        metrics.update(loss)

        if batch_idx == 0:
            result = self.model.bbox_decoder(preds)
            dt_image = self.draw_bbox(
                images[0],
                result["bboxes"][0],
                result["scores"][0],
                result["category_ids"][0],
            )
            gt_image = self.draw_bbox(
                images[0],
                labels[0]["boxes"],
                torch.ones(labels[0]["labels"].shape),
                labels[0]["labels"],
            )
            self.logger.experiment.add_image(
                "val/dt_image",
                dt_image,
                self.global_step,
            )
            self.logger.experiment.add_image(
                "val/gt_image",
                gt_image,
                self.global_step,
            )

        self.log_dict(metrics)

        if self.coco_eval is not None:
            result = self.model.bbox_decoder(preds)
            num_images = len(images)

            for i in range(num_images):
                image_id = torch.full_like(result["category_ids"][i], image_ids[i])
                self.coco_eval.update_dt(
                    image_id,
                    result["category_ids"][i],
                    result["bboxes"][i],
                    result["scores"][i],
                )

    def draw_bbox(self, single_image, single_bbox, single_score, single_category_id):
        image = []

        single_score = single_score.detach().cpu().tolist()
        single_category_id = single_category_id.detach().cpu().tolist()

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

        image += [single_image]

        return torch.cat(image, dim=2)

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


@register_task("detection")
def get_detection_task(task: Task, **kwargs):
    assert task.type == "detection"
    if "total_steps" in kwargs:
        task.detection.lr_scheduler.total_steps = kwargs["total_steps"]

    return DetectionTask(task.detection, **kwargs)
