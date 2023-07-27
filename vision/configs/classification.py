import dataclasses
from typing import Optional

from vision.configs import backbones, heads
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class ClassificationModel(ModelConfig):
    """ResNet config."""

    type: Optional[str] = "classification"

    backbone: backbones.Backbone = backbones.Backbone(type="alembic_resnet")
    head: heads.Head = heads.Head(type="classification")
