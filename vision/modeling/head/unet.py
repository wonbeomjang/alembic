from typing import Dict

import torch
from torch import nn, Tensor

from vision.configs import heads as head_config
from vision.utils.blocks import ConvReLUBN
from vision.modeling.head import register_head


class Unet(nn.Module):
    def __init__(self, config: head_config.Head, in_channels):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleDict()
        self.upsample = nn.Upsample(scale_factor=2)
        self.pyramid_levels = config.unet.pyramid_levels

        for i in range(len(self.pyramid_levels) - 1):
            self.blocks[self.pyramid_levels[i]] = nn.Sequential(
                ConvReLUBN(
                    in_channels[self.pyramid_levels[i]]
                    + in_channels[self.pyramid_levels[i + 1]],
                    in_channels[self.pyramid_levels[i]],
                    3,
                    1,
                    1,
                ),
                ConvReLUBN(
                    in_channels[self.pyramid_levels[i]],
                    in_channels[self.pyramid_levels[i]],
                    3,
                    1,
                    1,
                ),
            )

        self.blocks[self.pyramid_levels[-1]] = nn.Sequential(
            ConvReLUBN(
                in_channels[self.pyramid_levels[-1]],
                in_channels[self.pyramid_levels[-1]],
                3,
                1,
                1,
            ),
            ConvReLUBN(
                in_channels[self.pyramid_levels[-1]],
                in_channels[self.pyramid_levels[-1]],
                3,
                1,
                1,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvReLUBN(
                in_channels[self.pyramid_levels[0]],
                in_channels[self.pyramid_levels[0]],
                3,
                1,
                1,
            ),
            nn.Conv2d(in_channels[self.pyramid_levels[0]], config.num_classes, 1, 1),
        )

    def forward(self, x: Dict[str, Tensor]):
        for i in range(len(self.pyramid_levels) - 2, -1, -1):
            x[self.pyramid_levels[i + 1]] = self.upsample(x[self.pyramid_levels[i + 1]])
            x[self.pyramid_levels[i]] = torch.cat(
                (x[self.pyramid_levels[i]], x[self.pyramid_levels[i + 1]]), dim=1
            )
            x[self.pyramid_levels[i]] = self.blocks[self.pyramid_levels[i]](
                x[self.pyramid_levels[i]]
            )

        output = self.classifier(x[self.pyramid_levels[0]])
        return output


@register_head("unet")
def get_unet(config: head_config.Head, in_channels):
    assert config.type == "unet"

    return Unet(config, in_channels)
