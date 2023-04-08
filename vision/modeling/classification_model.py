import torch
from torch import Tensor
from torch import nn

from vision.configs import classification_model as classification_cfg
from vision.modeling.backbones import get_model
from vision.modeling import register_model


class ClassificationModel(nn.Module):
    def __init__(
        self,
        model_config: classification_cfg.ClassificationModel,
    ):
        super().__init__()
        self.backbone = get_model(model_config.backbone)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.randn((1, 3, 256, 256))
            output = self.backbone(dummy_input)
            self.max_key = max(output.keys())
            _, channels, _, _ = output[self.max_key].shape

        self.fc = nn.Linear(channels * 7 * 7, model_config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)[self.max_key]
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


@register_model("classification")
def classification(model_cfg: ClassificationModel):
    model = ClassificationModel(model_cfg)

    return model
