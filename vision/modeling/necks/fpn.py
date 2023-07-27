from collections import OrderedDict
from typing import Dict, List, Optional, Callable

from torch import nn, Tensor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import (
    ExtraFPNBlock,
    LastLevelMaxPool,
    LastLevelP6P7,
)

from vision.configs import necks as neck_config
from vision.modeling.necks import register_neck


class FPN(FeaturePyramidNetwork):
    def __init__(
        self,
        in_channels_list: List[int],
        pyramid_levels: List[str],
        out_channels: int = 256,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__(in_channels_list, out_channels, extra_blocks, norm_layer)
        self.pyramid_levels = pyramid_levels

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        res = OrderedDict()
        for i in self.pyramid_levels:
            res[str(i)] = x[str(i)]
        res = super().forward(res)
        return res


@register_neck("fpn")
def get_fpn(config: neck_config.Neck, in_channels_list: Optional[List[int]]):
    assert config.type == "fpn"

    if config.fpn.extra_blocks:
        extra_blocks = LastLevelMaxPool()
    else:
        extra_blocks = LastLevelP6P7(in_channels_list[-1], config.fpn.num_channels)

    return FPN(
        in_channels_list,
        config.fpn.pyramid_levels,
        config.fpn.num_channels,
        extra_blocks=extra_blocks,
    )
