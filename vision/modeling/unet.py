import torch
from torch import Tensor
from torch import nn

from vision.configs import unet as unet_cfg
from vision.configs.base_config import ModelConfig
from vision.modeling.backbones import get_backbone
from vision.modeling import register_model
from vision.utils.blocks import get_in_channels
from vision.modeling.head import get_head
from vision.modeling.necks import get_neck


class Unet(nn.Module):
    def __init__(
        self,
        model_config: unet_cfg.Unet,
    ):
        super().__init__()
        self.backbone = get_backbone(model_config.backbone)
        self.extra_block = nn.ModuleDict()

        in_channels = get_in_channels(self.backbone, self.extra_block, model_config.head.unet.min_level, model_config.head.unet.max_level)
                    
        model_config.head.unet.in_channels = in_channels
        model_config.head._num_classes = model_config.num_classes

        self.neck = get_neck(model_config.neck)
        self.head = get_head(model_config.head)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        for k, block in self.extra_block.items():
            x[k] = block(x[str(int(k) - 1)])
        x = self.neck(x)
        x = self.head(x)

        return x


@register_model("unet")
def unet(model_cfg: ModelConfig):
    assert isinstance(model_cfg, unet_cfg.Unet)

    model = Unet(model_cfg)

    return model


if __name__ == "__main__":
    model = Unet(unet_cfg.Unet())

    dummy_input = torch.randn((1, 3, 256, 256))
    x = model(dummy_input)

    print(x.shape)
