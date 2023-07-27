from typing import Optional, List

from torch import nn

from vision.configs import necks as neck_config
from vision.modeling.necks import register_neck


@register_neck("identity")
def identity(config: neck_config.Neck, in_channels_list: Optional[List[int]] = None):
    assert config.type == "identity"

    return nn.Identity(config)
