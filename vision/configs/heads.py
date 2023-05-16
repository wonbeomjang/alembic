import dataclasses
from typing import Optional, Tuple


@dataclasses.dataclass
class YOLO:
    num_blocks: int = 1
    num_channels: Optional[int] = None

    min_level: Optional[int] = None
    max_level: Optional[int] = None

    ratios: Tuple[float, float, float] = (0.5, 1.0, 1.5)
    scales: Tuple[float, float, float] = (
        2 ** (0.0 / 3.0),
        2 ** (1.0 / 3.0),
        2 ** (2.0 / 3.0),
    )


@dataclasses.dataclass
class Head:
    type: Optional[str] = None
    num_classes: int = 80
    yolo: YOLO = YOLO()
