import dataclasses
from typing import Optional

from vision.configs import backbones, necks, heads
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class Unet(ModelConfig):
    """ResNet config."""

    backbone: backbones.Backbone = backbones.Backbone(type="alembic_resnet")
    neck: necks.Neck = necks.Neck(type="identity")
    head: heads.Head = heads.Head(
        type="unet",
        unet=heads.Unet(),
    )


@dataclasses.dataclass
class Segmentation(ModelConfig):
    type: Optional[str] = "unet"
    unet: Unet = Unet()
