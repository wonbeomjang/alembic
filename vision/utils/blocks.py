from collections import OrderedDict
from typing import Union, Dict, List

import torch
from torch import nn


class ConvReLUBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


def get_in_channels(backbone: nn.Module) -> Dict[str, int]:
    in_channels = OrderedDict()

    with torch.no_grad():
        dummy_input = torch.randn((1, 3, 256, 256))
        output = backbone(dummy_input)

        for k in output.keys():
            in_channels[str(k)] = output[str(k)].size(1)

    return in_channels


def parse_in_channels(
    in_channels: Dict[str, int], pyramid_levels: List[str]
) -> Dict[str, int]:
    in_channels = {str(level): in_channels[str(level)] for level in pyramid_levels}
    return in_channels
