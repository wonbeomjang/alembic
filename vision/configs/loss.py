import dataclasses

try:
    from typing_extensions import Literal
    from typing import Optional
except ModuleNotFoundError:
    from typing import Optional, Literal

from torch import Tensor


@dataclasses.dataclass
class CrossEntropyLoss:
    weight: Optional[Tensor] = None
    size_average = None
    ignore_index: int = -100
    reduce = None
    reduction: str = "mean"
    label_smoothing: float = 0.0


@dataclasses.dataclass
class YOLOv4Loss:
    bg_iou_thresh: float = 0.4
    fg_iou_thresh: float = 0.5
    bbox_loss_type: Literal["l1", "smooth_l1", "ciou", "diou", "giou"] = "ciou"  # type: ignore


@dataclasses.dataclass
class Loss:
    type: Optional[str] = None
    cross_entropy_loss: CrossEntropyLoss = CrossEntropyLoss()
    yolo_v4_loss: YOLOv4Loss = YOLOv4Loss()
