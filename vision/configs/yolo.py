import dataclasses
from typing import Optional

from vision.configs import backbones, necks, heads
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class YOLO(ModelConfig):
    """ResNet config."""

    type: Optional[str] = "yolo"
    backbone: backbones.Backbone = backbones.Backbone(type="alembic_resnet")
    neck: necks.Neck = necks.Neck(type="fpn")
    head: heads.Head = heads.Head(type="yolo")

    head.yolo.num_channels = neck.fpn.num_channels
    head.yolo.min_level = neck.fpn.min_level
    head.yolo.max_level = neck.fpn.max_level
