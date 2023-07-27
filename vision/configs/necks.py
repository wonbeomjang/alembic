import dataclasses
from typing import Optional, Dict, List


@dataclasses.dataclass
class FPN:
    in_channels: Optional[Dict[str, int]] = None
    num_channels: int = 256

    pyramid_levels: List[str] = dataclasses.field(
        default_factory=lambda: ["2", "3", "4"]
    )

    extra_blocks: bool = False


@dataclasses.dataclass
class Identity:
    pass


@dataclasses.dataclass
class Neck:
    type: Optional[str] = None
    fpn: FPN = FPN()
    identity: Identity = Identity()
