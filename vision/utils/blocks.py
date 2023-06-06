from typing import Union

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


def get_in_channels(
    backbone: nn.Module, extra_block: nn.ModuleDict, min_level: int, max_level: int
):
    in_channels = {}

    with torch.no_grad():
        dummy_input = torch.randn((1, 3, 256, 256))
        output = backbone(dummy_input)
        for k in range(min_level, max_level + 1):
            if not str(k) in output.keys():
                c = in_channels[str(k - 1)]

                extra_block[str(k)] = nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    ConvReLUBN(c, 2 * c, 3, 1, 1),
                    ConvReLUBN(2 * c, 2 * c, 3, 1, 1),
                )
                in_channels[str(k)] = 2 * c
            else:
                in_channels[str(k)] = output[str(k)].size(1)

    return in_channels
