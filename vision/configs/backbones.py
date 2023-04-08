import dataclasses
from typing import Optional
from torchvision.models._api import WeightsEnum  # type: ignore


@dataclasses.dataclass
class AlembicResNet:
    """ResNet config."""

    model_id: str = "resnet18"
    weight: Optional[WeightsEnum] = None
    progress: bool = True


@dataclasses.dataclass
class Backbone:
    type: Optional[str] = None
    alembic_resnet: AlembicResNet = AlembicResNet()
