import dataclasses
from typing import Optional

from torch import Tensor


@dataclasses.dataclass
class CrossEntropyLoss:
    weight: Optional[Tensor] = None
    size_average = None
    ignore_index: int = -100
    reduce = None
    reduction: str = "mean"
    label_smoothing: float = 0.0


@dataclasses.dataclass
class Loss:
    type: Optional[str] = None
    cross_entropy_loss: CrossEntropyLoss = CrossEntropyLoss()
