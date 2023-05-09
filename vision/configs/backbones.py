import dataclasses
from typing import Optional, Union, Literal
from torchvision.models._api import WeightsEnum  # type: ignore
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class AlembicResNet(ModelConfig):
    """ResNet config."""

    model_id: str = "resnet18"
    progress: bool = True


@dataclasses.dataclass
class AlembicMobileNet(ModelConfig):
    """ResNet config."""

    model_id: str = "mobilenet_v2"
    progress: bool = True


@dataclasses.dataclass
class AlembicGhostNet(ModelConfig):
    """ResNet config."""

    model_id: str = "ghostnet_050"


@dataclasses.dataclass
class Backbone(ModelConfig):
    type: Optional[str] = None
    pretrained: bool = False
    alembic_resnet: AlembicResNet = AlembicResNet()
    alembic_mobilenet: AlembicMobileNet = AlembicMobileNet()
    alembic_ghostnet: AlembicGhostNet = AlembicGhostNet()
