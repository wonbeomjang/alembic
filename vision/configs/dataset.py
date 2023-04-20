import dataclasses
from typing import Tuple, Optional


@dataclasses.dataclass
class Dataset:
    type: Optional[str] = None
    image_dir: Optional[str] = None
    label_path: Optional[str] = None

    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 256
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""
