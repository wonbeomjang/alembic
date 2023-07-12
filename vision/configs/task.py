import dataclasses
from typing import Optional

try:
    from typing import Literal
except ModuleNotFoundError:
    from typing_extensions import Literal

from vision.configs.classification import ClassificationModel
from vision.configs.loss import Loss
from vision.configs.lr_scheduler import LRScheduler
from vision.configs.optimizer import Optimizer
from vision.configs.detection import DetectionModel


@dataclasses.dataclass
class ClassificationTask:
    model: ClassificationModel = ClassificationModel()
    initial_weight_path: Optional[str] = None
    initial_weight_type: Literal["full", "backbone"] = "full"  # type: ignore

    total_steps: Optional[int] = None
    optimizer: Optimizer = Optimizer(type="adam")
    lr_scheduler: Optional[LRScheduler] = None
    loss: Optional[Loss] = Loss(type="cross_entropy_loss")
    task: Literal["binary", "multiclass", "multilabel"] = "multiclass"  # type: ignore


@dataclasses.dataclass
class DetectionTask:
    model: DetectionModel = DetectionModel()
    initial_weight_path: Optional[str] = None
    initial_weight_type: Literal["full", "backbone"] = "full"  # type: ignore

    total_steps: Optional[int] = None
    optimizer: Optimizer = Optimizer(type="adam")
    lr_scheduler: Optional[LRScheduler] = None
    loss: Optional[Loss] = Loss(type="yolo_v4_loss")


@dataclasses.dataclass
class Task:
    type: Optional[str] = None
    classification: ClassificationTask = ClassificationTask()
    detection: DetectionTask = DetectionTask()
