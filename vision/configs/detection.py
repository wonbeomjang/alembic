import dataclasses
from typing import Optional

from vision.configs import backbones, necks, heads
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class YOLO(ModelConfig):
    """ResNet config."""

    type: Optional[str] = "yolo"
    backbone: backbones.Backbone = backbones.Backbone(type="alembic_mobilenet")
    neck: necks.Neck = necks.Neck(type="fpn")
    head: heads.Head = heads.Head(type="yolo")

    iou_threshold: float = 0.5

    head.yolo.num_channels = neck.fpn.num_channels


@dataclasses.dataclass
class DetectionModel(ModelConfig):
    type: Optional[str] = None
    num_classes: Optional[int] = None

    yolo: YOLO = YOLO()
