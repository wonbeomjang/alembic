from typing import Dict

import torch
from torch import nn, Tensor

from vision.configs import yolo as yolo_cfg
from vision.configs.base_config import ModelConfig
from vision.modeling import register_model
from vision.modeling.backbones import get_backbone
from vision.modeling.head import get_head
from vision.modeling.necks import get_neck


class YOLO(nn.Module):
    def __init__(
        self,
        model_config: yolo_cfg.YOLO,
    ):
        super().__init__()
        self.backbone = get_backbone(model_config.backbone)

        with torch.no_grad():
            dummy_input = torch.randn((1, 3, 224, 224))
            output = self.backbone(dummy_input)

        in_channels = {}
        for k in output.keys():
            in_channels[k] = output[k].size(1)

        model_config.neck.fpn.in_channels = in_channels

        self.neck = get_neck(model_config.neck)
        self.head = get_head(model_config.head)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor]]:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


@register_model("yolo")
def classification(model_cfg: ModelConfig):
    assert isinstance(model_cfg, yolo_cfg.YOLO)
    assert model_cfg.head.yolo.min_level == model_cfg.neck.fpn.min_level
    assert model_cfg.head.yolo.max_level == model_cfg.neck.fpn.max_level

    model = YOLO(model_cfg)

    return model
