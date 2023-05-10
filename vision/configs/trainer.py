import dataclasses

try:
    from typing_extensions import Literal
    from typing import Optional
except ModuleNotFoundError:
    from typing import Optional, Literal

from vision.configs.classification import ClassificationModel
from vision.configs.loss import Loss
from vision.configs.lr_scheduler import LRScheduler
from vision.configs.optimizer import Optimizer
from vision.configs.dataset import Dataset


@dataclasses.dataclass
class ClassificationTrainer:
    classification_model: ClassificationModel = ClassificationModel()
    total_steps: Optional[int] = None
    optimizer: Optimizer = Optimizer(type="adam")
    lr_scheduler: Optional[LRScheduler] = None
    loss: Loss = Loss(type="cross_entropy_loss")
    task: Literal["binary", "multiclass", "multilabel"] = "multiclass"  # type: ignore


@dataclasses.dataclass
class Trainer:
    type: Optional[str] = None
    classification: ClassificationTrainer = ClassificationTrainer()
    epochs: int = 0
    train_data: Dataset = Dataset()
    val_data: Optional[Dataset] = None

    logger: Literal["tensorboard", None] = None  # type: ignore
    log_dir: str = "logs"
    save_best_model: bool = False
    save_model: bool = False
    ckpt: Optional[str] = None
