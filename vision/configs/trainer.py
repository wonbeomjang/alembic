import dataclasses
from typing import Optional
from typing_extensions import Literal

from vision.configs.classification import ClassificationModel
from vision.configs.loss import Loss
from vision.configs.optimizer import Optimizer
from vision.configs.dataset import Dataset


@dataclasses.dataclass
class ClassificationTrainer:
    classification_model: ClassificationModel = ClassificationModel()
    optimizer: Optimizer = Optimizer(type="adam")
    loss: Loss = Loss(type="cross_entropy_loss")
    task: Literal["binary", "multiclass", "multilabel"] = "multiclass"


@dataclasses.dataclass
class Trainer:
    type: Optional[str] = None
    classification: ClassificationTrainer = ClassificationTrainer()
    epochs: int = 0
    train_data: Dataset = Dataset()
    val_data: Optional[Dataset] = None
