import dataclasses
from typing import Optional

from vision.configs import backbones, necks, heads
from vision.configs.base_config import ModelConfig


@dataclasses.dataclass
class Unet(ModelConfig):
    """ResNet config."""

    type: Optional[str] = "unet"
    num_classes: int = 2
    backbone: backbones.Backbone = backbones.Backbone(type="alembic_resnet")
    neck: necks.Neck = necks.Neck(type="identity")
    head: heads.Head = heads.Head(
        type="unet",
        unet=heads.Unet(
            min_level=0,
            max_level=4,
        )
    )
