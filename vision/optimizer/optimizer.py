from torch.optim.adam import Adam

from vision.configs.optimizer import Optimizer
from vision.optimizer import register_optimizer


@register_optimizer("adam")
def adam(optimizer_config: Optimizer):
    assert optimizer_config.type == "adam"

    def _adam(param):
        return Adam(
            param,
            lr=optimizer_config.lr,
            betas=optimizer_config.adam.betas,
            eps=optimizer_config.adam.eps,
            weight_decay=optimizer_config.adam.weight_decay,
            amsgrad=optimizer_config.adam.amsgrad,
            foreach=optimizer_config.adam.foreach,
            maximize=optimizer_config.adam.maximize,
            capturable=optimizer_config.adam.capturable,
            differentiable=optimizer_config.adam.differentiable,
            fused=optimizer_config.adam.fused,
        )

    return _adam
