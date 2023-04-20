import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfig:
    type: Optional[str] = None
