import dataclasses
from typing import Optional

from vision.configs import backbones
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class ClassificationModel(ModelConfig):
    """ResNet config."""

    type: Optional[str] = "classification"
    num_classes: Optional[int] = None
    backbone = backbones.Backbone(type="alembic_resnet")
