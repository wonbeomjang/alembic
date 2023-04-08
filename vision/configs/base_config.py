import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfig:
    model_id: Optional[str] = None
