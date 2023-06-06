import dataclasses
from typing import Optional, Dict


@dataclasses.dataclass
class FPN:
    in_channels: Optional[Dict[str, int]] = None
    num_channels: int = 256

    min_level: int = 2
    max_level: int = 4


@dataclasses.dataclass
class Identity:
    pass


@dataclasses.dataclass
class Neck:
    type: Optional[str] = None
    fpn: FPN = FPN()
    identity: Identity = Identity()
