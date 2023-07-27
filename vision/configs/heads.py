import dataclasses
from typing import Optional, Tuple, List


@dataclasses.dataclass
class Classification:
    pass


@dataclasses.dataclass
class YOLO:
    num_blocks: int = 3

    ratios: Tuple[float, float, float] = (0.5, 1.0, 1.5)
    scales: Tuple[float, float, float] = (
        2 ** (0.0 / 3.0),
        2 ** (1.0 / 3.0),
        2 ** (2.0 / 3.0),
    )


@dataclasses.dataclass
class Unet:
    pyramid_levels: List[str] = dataclasses.field(
        default_factory=lambda: ["0", "1", "2", "3", "4"]
    )


@dataclasses.dataclass
class Head:
    type: Optional[str] = None
    num_classes: Optional[int] = None
    yolo: YOLO = YOLO()
    unet: Unet = Unet()
