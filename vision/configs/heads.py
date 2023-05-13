import dataclasses
from typing import Optional


@dataclasses.dataclass
class YOLO:
    num_blocks: int = 1
    num_channels: Optional[int] = None

    min_level: Optional[int] = None
    max_level: Optional[int] = None


@dataclasses.dataclass
class Head:
    type: Optional[str] = None
    num_classes: int = 80
    yolo: YOLO = YOLO()
