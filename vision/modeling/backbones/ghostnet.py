from timm import create_model
from torch import nn
from torchvision.models._api import register_model  # type: ignore
from torchvision.models.feature_extraction import create_feature_extractor
import timm

from vision.configs import backbones


__all__ = [
    "alembic_ghostnet",
]

_ghostnet_return_nodes = {
    "blocks.0": "0",
    "blocks.2": "1",
    "blocks.4": "2",
    "blocks.6": "3",
    "blocks.9": "4",
}

support_model = [
    "ghostnet_050",
    "ghostnet_100",
    "ghostnet_130",
]


@register_model("alembic_ghostnet")
def alembic_ghostnet(
    backbone_cfg: backbones.Backbone = backbones.Backbone(),
) -> nn.Module:
    """
    Support model id
    - mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

    :param backbone_cfg: config object of resnet
    :return: feature extraction model
    """
    assert backbone_cfg.type == "alembic_ghostnet"
    assert backbone_cfg.alembic_ghostnet.model_id in support_model

    model = create_model(
        backbone_cfg.alembic_ghostnet.model_id,
        pretrained=backbone_cfg.alembic_ghostnet.pretrained,
    )
    model = create_feature_extractor(
        model,
        return_nodes=_ghostnet_return_nodes,
    )

    return model
