from typing import Dict, Optional, Sequence

from vision.loss import register_loss

try:
    from typing import Literal
except ModuleNotFoundError:
    from typing_extensions import Literal

import torch
from torch import Tensor, nn
from torchvision.models.detection._utils import Matcher, BoxCoder
from torchvision.ops import boxes as box_ops, sigmoid_focal_loss
from vision.loss.utils import box_loss as box_criterion

from vision.configs import loss as loss_config


class YOLOV4Loss(nn.Module):
    def __init__(
        self,
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        bbox_loss_type: Literal["l1", "smooth_l1", "ciou", "diou", "giou"],  # type: ignore
        box_coder: BoxCoder,
        proposer_matcher: Optional[Matcher] = None,
    ):
        super().__init__()
        if proposer_matcher is None:
            proposer_matcher = Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.bbox_loss_type = bbox_loss_type
        self.proposer_matcher = proposer_matcher
        self.class_loss_weight = 1.0
        self.box_loss_weight = 1.0
        self.box_coder = box_coder

    def compute_loss(
        self, device, pred_bboxes, pred_labels, reshaped_anchor, targets
    ) -> Dict[str, Tensor]:
        class_losses = []
        box_losses = []
        for dt_label, dt_bbox, target, anchor in zip(
            pred_labels, pred_bboxes, targets, reshaped_anchor
        ):
            if target["boxes"].numel() == 0:
                continue
            match_metrix = box_ops.box_iou(target["boxes"], anchor)
            match_index: Tensor = self.proposer_matcher(match_metrix)

            foreground_index = match_index >= 0
            num_foreground = foreground_index.sum()
            valid_index = match_index != self.proposer_matcher.BETWEEN_THRESHOLDS

            gt_label = torch.zeros_like(dt_label, device=device)
            gt_label[
                foreground_index, target["labels"][match_index[foreground_index]]
            ] = 1

            # num_valid, num_classes
            # include background
            gt_label = gt_label[valid_index]
            dt_label = dt_label[valid_index]

            # num_foreground, 4
            gt_bbox = target["boxes"][match_index[foreground_index]]
            dt_bbox = dt_bbox[foreground_index, :]
            dt_anchor = anchor[foreground_index, :]

            class_loss = sigmoid_focal_loss(dt_label, gt_label, reduction="sum") / max(
                1, num_foreground
            )
            box_loss = box_criterion(
                self.bbox_loss_type, self.box_coder, dt_anchor, gt_bbox, dt_bbox
            )

            class_losses += [class_loss]
            box_losses += [box_loss]

        class_losses = sum(class_losses) / max(1, len(class_losses))
        box_losses = sum(box_losses) / max(1, len(box_losses))
        return {"class": class_losses, "box": box_losses}

    def forward(
        self,
        preds: Dict[str, Tensor],
        targets: Sequence[Tensor],
        anchor: Tensor,
    ) -> Dict[str, Tensor]:
        device = anchor.device
        loss = self.compute_loss(
            device, preds["boxes"], preds["labels"], anchor, targets
        )
        return loss


@register_loss("yolo_v4_loss")
def yolo_v4_loss(config: loss_config.Loss, **kwargs):
    assert config.type == "yolo_v4_loss"

    criterion = YOLOV4Loss(
        config.yolo_v4_loss.fg_iou_thresh,
        config.yolo_v4_loss.bg_iou_thresh,
        config.yolo_v4_loss.bbox_loss_type,
        **kwargs
    )
    return criterion
