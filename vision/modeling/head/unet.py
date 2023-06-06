from typing import Dict

import torch
from torch import nn, Tensor

from vision.configs import heads as head_config
from vision.utils.blocks import ConvReLUBN
from vision.modeling.head import register_head


class Unet(nn.Module):
    def __init__(self, config: head_config.Head):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleDict()
        self.upsample = nn.Upsample(scale_factor=2)
        for pyramid_level in range(config.unet.max_level - 1, config.unet.min_level - 1, -1):
            self.blocks[str(pyramid_level)] = nn.Sequential(
                ConvReLUBN(
                    config.unet.in_channels[str(pyramid_level)] + config.unet.in_channels[str(pyramid_level + 1)],
                    config.unet.in_channels[str(pyramid_level)],
                    3,
                    1,
                    1,
                ),
                ConvReLUBN(
                    config.unet.in_channels[str(pyramid_level)],
                    config.unet.in_channels[str(pyramid_level)],
                    3,
                    1,
                    1,
                )
            )

        pyramid_level = config.unet.min_level
        self.classifier = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvReLUBN(
                config.unet.in_channels[str(pyramid_level)],
                config.unet.in_channels[str(pyramid_level)],
                3,
                1,
                1,
            ),
            nn.Conv2d(
                config.unet.in_channels[str(pyramid_level)],
                config._num_classes,
                1,
                1
            )
        )

    def forward(self, x: Dict[str, Tensor]):
        for pyramid_level in range(
            self.config.unet.max_level - 1, self.config.unet.min_level - 1, -1
        ):
            x[str(pyramid_level + 1)] = self.upsample(x[str(pyramid_level + 1)])
            x[str(pyramid_level)] = torch.cat((x[str(pyramid_level + 1)], x[str(pyramid_level)]), dim=1)
            x[str(pyramid_level)] = self.blocks[str(pyramid_level)](x[str(pyramid_level)])
        output = self.classifier(x[str(self.config.unet.min_level)])
        return output


@register_head("unet")
def get_unet(config: head_config.Head):
    assert config.type == "unet"

    return Unet(config)
