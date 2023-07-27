from torch import Tensor
from torch import nn

from vision.configs import classification as classification_cfg
from vision.configs.base_config import ModelConfig
from vision.modeling.backbones import get_backbone
from vision.modeling import register_model
from vision.modeling.head import get_head
from vision.utils.blocks import get_in_channels


class ClassificationModel(nn.Module):
    def __init__(
        self,
        model_config: classification_cfg.ClassificationModel,
    ):
        super().__init__()
        self.backbone = get_backbone(model_config.backbone)

        channels = get_in_channels(self.backbone)
        self.head = get_head(model_config.head, channels[max(channels.keys())])

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)

        return x


@register_model("classification")
def classification(model_cfg: ModelConfig, **kwargs):
    assert isinstance(model_cfg, classification_cfg.ClassificationModel)

    model = ClassificationModel(model_cfg)

    return model
