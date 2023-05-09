from torch import nn
from torchvision.models._api import register_model  # type: ignore
from torchvision.models import get_model
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import mobilenet

from vision.configs import backbones


__all__ = [
    "alembic_mobilenet",
]

_torchvision_return_nodes = {
    "mobilenet_v2": {
        "features.1": "0",
        "features.3": "1",
        "features.6": "2",
        "features.13": "3",
        "features.18": "4",
    },
    "mobilenet_v3_small": {
        "features.0": "0",
        "features.1": "1",
        "features.3": "2",
        "features.8": "3",
        "features.12": "4",
    },
    "mobilenet_v3_large": {
        "features.1": "0",
        "features.3": "1",
        "features.6": "2",
        "features.12": "3",
        "features.16": "4",
    },
}

support_model = [
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
]

_weight_dict = {
    "mobilenet_v2": mobilenet.MobileNet_V2_Weights.IMAGENET1K_V1,
    "mobilenet_v3_small": mobilenet.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    "mobilenet_v3_large": mobilenet.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
}


@register_model("alembic_mobilenet")
def alembic_mobilenet(
    backbone_cfg: backbones.Backbone = backbones.Backbone(),
) -> nn.Module:
    """
    Support model id
    - mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

    :param backbone_cfg: config object of mobilenet
    :return: feature extraction model
    """
    assert backbone_cfg.type == "alembic_mobilenet"
    assert backbone_cfg.alembic_ghostnet.model_id in support_model

    if backbone_cfg.pretrained:
        weight = _weight_dict[backbone_cfg.alembic_mobilenet.model_id]
    else:
        weight = None

    model = get_model(
        backbone_cfg.alembic_mobilenet.model_id,
        weights=weight,
        progress=backbone_cfg.alembic_mobilenet.progress,
    )
    model = create_feature_extractor(
        model,
        return_nodes=_torchvision_return_nodes[backbone_cfg.alembic_mobilenet.model_id],
    )

    return model
