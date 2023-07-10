from torch import nn
from torchvision.models._api import register_model  # type: ignore

from vision.configs import backbones


__all__ = [
    "alembic_repvgg",
]

from vision.modeling.backbones.impl.repvgg import rep_vgg

support_model = [
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
    "repvgg_b0",
    "repvgg_b1",
    "repvgg_b2",
    "repvgg_b3",
]


@register_model("alembic_repvgg")
def alembic_repvgg(
    backbone_cfg: backbones.Backbone = backbones.Backbone(),
) -> nn.Module:
    """
    Support model id
    - mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

    :param backbone_cfg: config object of resnet
    :return: feature extraction model
    """
    assert backbone_cfg.type == "alembic_repvgg"
    assert backbone_cfg.alembic_repvgg.model_id in support_model

    model = rep_vgg(
        backbone_cfg.alembic_repvgg.model_id,
        pretrained=backbone_cfg.pretrained,
    )

    return model
