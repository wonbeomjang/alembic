import dataclasses
from typing import Literal, Optional, Tuple

from vision.configs.classification import ClassificationModel
from vision.configs.loss import Loss
from vision.configs.optimizer import Optimizer


@dataclasses.dataclass
class ClassificationTrainer:
    classification_model: ClassificationModel = ClassificationModel()
    optimizer: Optimizer = Optimizer(type="adam")
    loss: Loss = Loss(type="cross_entropy_loss")
    task: Literal["binary", "multiclass", "multilabel"] = "multiclass"


@dataclasses.dataclass
class DataConfig:
    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 256


@dataclasses.dataclass
class Trainer:
    type: Optional[str] = None
    classification: ClassificationTrainer = ClassificationTrainer()
    epochs: int = 0
