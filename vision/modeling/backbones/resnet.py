from torch import nn
from torchvision.models._api import register_model  # type: ignore
from torchvision.models import get_model, resnet
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

_weight_dict = {
    "resnet18": resnet.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": resnet.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": resnet.ResNet50_Weights.IMAGENET1K_V1,
    "resnet101": resnet.ResNet101_Weights.IMAGENET1K_V1,
    "resnet152": resnet.ResNet152_Weights.IMAGENET1K_V1,
    "resnext50_32x4d": resnet.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
    "resnext101_32x8d": resnet.ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
    "resnext101_64x4d": resnet.ResNeXt101_64X4D_Weights.IMAGENET1K_V1,
    "wide_resnet50_2": resnet.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
    "wide_resnet101_2": resnet.Wide_ResNet101_2_Weights.IMAGENET1K_V1,
}


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
    assert backbone_cfg.alembic_resnet.model_id in support_model

    if backbone_cfg.pretrained:
        weight = _weight_dict[backbone_cfg.alembic_mobilenet.model_id]
    else:
        weight = None

    model = get_model(
        backbone_cfg.alembic_resnet.model_id,
        weights=weight,
        progress=backbone_cfg.alembic_resnet.progress,
    )
    model = create_feature_extractor(model, return_nodes=_torchvision_return_nodes)

    return model
