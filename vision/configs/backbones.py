import dataclasses
from typing import Optional
from torchvision.models._api import WeightsEnum  # type: ignore
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class AlembicResNet(ModelConfig):
    """ResNet config."""

    model_id: str = "resnet18"
    weight: Optional[WeightsEnum] = None
    progress: bool = True


@dataclasses.dataclass
class Backbone(ModelConfig):
    type: Optional[str] = None
    alembic_resnet: AlembicResNet = AlembicResNet()
