import dataclasses
import os.path
from typing import Optional

from vision.configs.dataset import Dataset

try:
    from typing import Literal
except ModuleNotFoundError:
    from typing_extensions import Literal

from vision.configs.task import Task


@dataclasses.dataclass
class ExperimentConfig:
    task: Task = Task()

    train_data: Dataset = Dataset()
    val_data: Optional[Dataset] = None

    epochs: int = 100

    logger: Literal["tensorboard", None] = None  # type: ignore
    log_dir: str = os.path.join(".", "logs")
    run_name: str = "run"
    save_best_model: bool = True
    load_last_weight: bool = True
    auto_num_classes: bool = True
