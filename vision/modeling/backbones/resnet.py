from torch import nn
from torchvision.models._api import register_model  # type: ignore
from torchvision.models import get_model
from torchvision.models.feature_extraction import create_feature_extractor

from vision.configs import backbones


__all__ = [
    "alembic_resnet",
]

_torchvision_return_nodes = {
    "maxpool": "0",
    "layer1": "1",
    "layer2": "2",
    "layer3": "3",
    "layer4": "4",
}


support_model = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


@register_model("alembic_resnet")
def alembic_resnet(
    backbone_cfg: backbones.Backbone = backbones.Backbone(),
) -> nn.Module:
    """
    Support model id
    - resnet18, resnet34, resnet50, resnet101, resnet152, resnet152,
    - resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, resnext101_64x4d,
    - wide_resnet50_2, wide_resnet101_2

    :param backbone_cfg: config object of resnet
    :return: feature extraction model
    """
    assert backbone_cfg.type == "alembic_resnet"

    model = get_model(
        backbone_cfg.alembic_resnet.model_id,
        weights=backbone_cfg.alembic_resnet.weight,
        progress=backbone_cfg.alembic_resnet.progress,
    )
    model = create_feature_extractor(model, return_nodes=_torchvision_return_nodes)

    return model
