import dataclasses
from typing import Optional, Union, List


@dataclasses.dataclass
class OneCycleLR:
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: Union[float, List[float]] = 0.85
    max_momentum: Union[float, List[float]] = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    three_phase: bool = False
    last_epoch: int = -1
    verbose: bool = False


@dataclasses.dataclass
class LRScheduler:
    type: Optional[str] = None
    one_cycle_lr: OneCycleLR = OneCycleLR()

    total_steps: Optional[int] = None
