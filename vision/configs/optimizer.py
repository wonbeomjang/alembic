import dataclasses
from typing import Optional, Tuple


@dataclasses.dataclass
class Adam:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False
    foreach: Optional[bool] = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: Optional[bool] = None


@dataclasses.dataclass
class Optimizer:
    type: Optional[str] = None
    adam: Adam = Adam()
