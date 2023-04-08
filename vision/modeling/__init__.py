from typing import Any

import torchvision
from torch import nn

from vision.modeling import backbones
from vision.configs import backbones as backbone_cfg


def get_model(backbone_config: backbone_cfg.Backbone, **config: Any) -> nn.Module:
    return torchvision.models.get_model(
        name=backbone_config.type, backbone_cfg=backbone_config, **config
    )
