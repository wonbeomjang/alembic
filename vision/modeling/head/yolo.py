from typing import Dict

import torch
from torch import nn, Tensor

from vision.configs import heads as head_config
from vision.modeling.head import register_head


class YOLOHead(nn.Module):
    def __init__(self, config: head_config.Head, in_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.num_anchors = len(config.yolo.ratios) * len(config.yolo.scales)
        self.head = nn.Conv2d(
            in_channels,
            self.num_anchors * (4 + 1 + config.num_classes),
            3,
            1,
            1,
        )

    def forward(self, x: Dict[str, Tensor]):
        result = []

        for pyramid_level in x.keys():
            res = self.head(x[pyramid_level])
            n, _, w, h = res.shape
            result += [res.reshape(n, self.num_anchors * w * h, -1)]
        result = torch.cat(result, dim=1)

        output = {"boxes": result[..., :4], "labels": result[..., 4:]}

        return output


@register_head("yolo")
def yolo_head(config: head_config.Head, in_channels: int):
    assert config.type == "yolo"

    return YOLOHead(config, in_channels)
