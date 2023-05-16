from collections import defaultdict
from typing import Dict

from torch import nn, Tensor

from vision.configs import heads as head_config
from vision.modeling.head import register_head


class YOLOHead(nn.Module):
    def __init__(self, config: head_config.Head):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.num_anchors = len(config.yolo.ratios) * len(config.yolo.scales)
        self.min_level = config.yolo.min_level
        self.max_level = config.yolo.max_level

        head = {}
        for pyramid_level in range(config.yolo.min_level, config.yolo.max_level + 1):
            _head = []
            for i in range(config.yolo.num_blocks - 1):
                _head += [
                    nn.Conv2d(config.yolo.num_channels, config.yolo.num_channels, 3, 1)
                ]

            _head += [
                nn.Conv2d(
                    config.yolo.num_channels,
                    self.num_anchors * (4 + 1 + config.num_classes),
                    3,
                    1,
                )
            ]

            head[str(pyramid_level)] = nn.Sequential(*_head)
        self.head = head

    def forward(self, x: Dict[str, Tensor]):
        output = defaultdict(dict)

        for pyramid_level in range(self.min_level, self.max_level + 1):
            res = self.head[str(pyramid_level)](x[str(pyramid_level)])
            n, _, h, w = res.shape

            res = res.reshape(n, self.num_anchors, -1, h, w)

            output[str(pyramid_level)]["boxes"] = res[:, :, :4, :, :]
            output[str(pyramid_level)]["background"] = res[:, :, 4:5, :, :]
            output[str(pyramid_level)]["classes"] = res[:, :, 5:, :, :]

        return output


@register_head("yolo")
def yolo_head(config: head_config.Head):
    assert config.type == "yolo"

    return YOLOHead(config)
