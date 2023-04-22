from typing import Callable

from torch import optim
from torch.optim import lr_scheduler

from vision.lr_scheduler import register_lr_scheduler
from vision.configs.lr_scheduler import LRScheduler


@register_lr_scheduler("one_cycle_lr")
def one_cycle_lr(config: LRScheduler) -> Callable[..., optim.lr_scheduler.LRScheduler]:
    assert config.type == "one_cycle_lr"

    def _one_cycle_lr(optimizer: optim.Optimizer, max_lr: float):
        return lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=config.one_cycle_lr.total_steps,
            epochs=config.one_cycle_lr.epochs,
            steps_per_epoch=config.one_cycle_lr.steps_per_epoch,
            pct_start=config.one_cycle_lr.pct_start,
            anneal_strategy=config.one_cycle_lr.anneal_strategy,
            cycle_momentum=config.one_cycle_lr.cycle_momentum,
            base_momentum=config.one_cycle_lr.base_momentum,
            max_momentum=config.one_cycle_lr.max_momentum,
            div_factor=config.one_cycle_lr.div_factor,
            final_div_factor=config.one_cycle_lr.final_div_factor,
            three_phase=config.one_cycle_lr.three_phase,
            last_epoch=config.one_cycle_lr.last_epoch,
            verbose=config.one_cycle_lr.verbose,
        )

    return _one_cycle_lr
