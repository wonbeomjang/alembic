import dataclasses
from typing import Optional, Union
from torchvision.models._api import WeightsEnum  # type: ignore
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class AlembicResNet(ModelConfig):
    """ResNet config."""

    model_id: str = "resnet18"
    weight: Optional[Union[WeightsEnum, str]] = None
    progress: bool = True


@dataclasses.dataclass
class AlembicMobileNet(ModelConfig):
    """ResNet config."""

    model_id: str = "mobilenet_v3_large"
    weight: Optional[Union[WeightsEnum, str]] = None
    progress: bool = True


@dataclasses.dataclass
class Backbone(ModelConfig):
    type: Optional[str] = None
    alembic_resnet: AlembicResNet = AlembicResNet()
    alembic_mobilenet: AlembicMobileNet = AlembicMobileNet()
