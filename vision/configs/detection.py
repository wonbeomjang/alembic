import dataclasses
from typing import Optional

from vision.configs import backbones, necks, heads
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class YOLO(ModelConfig):
    """ResNet config."""

    backbone: backbones.Backbone = backbones.Backbone(type="alembic_resnet")
    neck: necks.Neck = necks.Neck(type="fpn")
    head: heads.Head = heads.Head(type="yolo")

    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.4

    head.yolo.num_channels = neck.fpn.num_channels


@dataclasses.dataclass
class DetectionModel(ModelConfig):
    type: Optional[str] = None
    num_classes: Optional[int] = None

    yolo: YOLO = YOLO()
