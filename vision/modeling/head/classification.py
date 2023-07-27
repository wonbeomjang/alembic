from typing import Dict

from torch import nn, Tensor

from vision.configs import heads as head_config
from vision.modeling.head import register_head


class ClassificationHead(nn.Module):
    def __init__(self, config: head_config.Head, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(in_channels * 8 * 8, config.num_classes)

    def forward(self, x: Dict[str, Tensor]):
        x = self.avg_pool(x.popitem()[-1])
        x = self.flatten(x)
        x = self.head(x)
        return x


@register_head("classification")
def yolo_head(config: head_config.Head, in_channels: int):
    assert config.type == "classification"

    return ClassificationHead(config, in_channels)
