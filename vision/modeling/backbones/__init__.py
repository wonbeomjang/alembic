from typing import Any

import torchvision
from torch import nn

from vision.configs import backbones as backbone_cfg
from vision.modeling.backbones import impl
from vision.modeling.backbones import resnet, mobilenet, ghostnet, repvgg


def get_backbone(backbone_config: backbone_cfg.Backbone, **config: Any) -> nn.Module:
    return torchvision.models.get_model(
        name=backbone_config.type, backbone_cfg=backbone_config, **config
    )
