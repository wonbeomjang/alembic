from collections import OrderedDict
from typing import Dict

from torch import nn, Tensor

from vision.configs import necks as neck_config
from vision.modeling.necks import register_neck


class FPN(nn.Module):
    def __init__(self, config: neck_config.Neck):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleDict()
        self.upsample = nn.Upsample(scale_factor=2)
        for pyramid_level in range(config.fpn.max_level, config.fpn.min_level - 1, -1):
            self.blocks[str(pyramid_level)] = nn.Conv2d(
                config.fpn.in_channels[str(pyramid_level)],
                config.fpn.num_channels,
                3,
                1,
                1,
            )

    def forward(self, x: Dict[str, Tensor]):
        output = OrderedDict()

        for pyramid_level in range(
            self.config.fpn.min_level, self.config.fpn.max_level + 1
        ):
            output[str(pyramid_level)] = self.blocks[str(pyramid_level)](
                x[str(pyramid_level)]
            )

        for pyramid_level in range(
            self.config.fpn.max_level - 1, self.config.fpn.min_level - 1, -1
        ):
            up_sampled = self.upsample(output[str(pyramid_level + 1)])
            output[str(pyramid_level)] = up_sampled + output[str(pyramid_level)]

        return output


@register_neck("fpn")
def get_fpn(config: neck_config.Neck):
    assert config.type == "fpn"

    return FPN(config)
