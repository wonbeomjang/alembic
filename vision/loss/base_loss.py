from torch import nn

from vision.loss import register_loss
from vision.configs import loss as loss_config


@register_loss("cross_entropy_loss")
def cross_entropy_loss(config: loss_config.Loss):
    assert config.type == "cross_entropy_loss"

    criterion = nn.CrossEntropyLoss(
        weight=config.cross_entropy_loss.weight,
        size_average=config.cross_entropy_loss.size_average,
        ignore_index=config.cross_entropy_loss.ignore_index,
        reduce=config.cross_entropy_loss.reduce,
        reduction=config.cross_entropy_loss.reduction,
        label_smoothing=config.cross_entropy_loss.label_smoothing,
    )
    return criterion
