import dataclasses
from typing import Optional
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class AlembicResNet(ModelConfig):
    """ResNet config."""

    model_id: str = "resnet18"
    progress: bool = True


@dataclasses.dataclass
class AlembicMobileNet(ModelConfig):
    """ResNet config."""

    model_id: str = "mobilenet_v3_small"
    progress: bool = True


@dataclasses.dataclass
class AlembicGhostNet(ModelConfig):
    """ResNet config."""

    model_id: str = "ghostnetv2_100"


@dataclasses.dataclass
class Backbone(ModelConfig):
    type: Optional[str] = None
    pretrained: bool = True
    alembic_resnet: AlembicResNet = AlembicResNet()
    alembic_mobilenet: AlembicMobileNet = AlembicMobileNet()
    alembic_ghostnet: AlembicGhostNet = AlembicGhostNet()
