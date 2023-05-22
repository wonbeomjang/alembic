from typing import Dict, List, Optional

from vision.loss import register_loss

try:
    from typing_extensions import Literal
except ModuleNotFoundError:
    from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection._utils import Matcher
from torchvision.ops import boxes as box_ops, sigmoid_focal_loss
from vision.loss.utils import box_loss as box_criterion

from vision.configs import loss as loss_config


class YOLOV4Loss(nn.Module):
    def __init__(
        self,
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        bbox_loss_type: Literal["l1", "smooth_l1", "ciou", "diou", "giou"],
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
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.class_loss_weight = 1.0
        self.box_loss_weight = 1.0

    def convert_preds_format(self, anchor, num_classes, num_images, preds):
        pred_bboxes = []
        pred_labels = []
        reshaped_anchor = torch.cat(
            [v.reshape(num_images, -1, 4) for v in anchor.values()], dim=1
        )
        for pred in preds.values():
            pred_bboxes += [pred["boxes"].reshape(num_images, -1, 4)]
            pred_labels += [pred["labels"].reshape(num_images, -1, num_classes)]
        pred_bboxes = torch.cat(pred_bboxes, dim=1)
        pred_labels = torch.cat(pred_labels, dim=1)
        return pred_bboxes, pred_labels, reshaped_anchor

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
        preds: Dict[str, Dict[str, Tensor]],
        targets: List[Dict[str, Tensor]],
        anchor: Dict[str, Tensor],
    ) -> Tensor:
        num_images = len(targets)
        num_classes = preds[min(preds.keys())]["labels"].shape[-1]
        device = anchor[min(anchor.keys())].device

        pred_bboxes, pred_labels, reshaped_anchor = self.convert_preds_format(
            anchor, num_classes, num_images, preds
        )
        loss = self.compute_loss(
            device, pred_bboxes, pred_labels, reshaped_anchor, targets
        )
        return loss


@register_loss("yolo_v4_loss")
def yolo_v4_loss(config: loss_config.Loss):
    assert config.type == "yolo_v4_loss"

    criterion = YOLOV4Loss(
        config.yolo_v4_loss.fg_iou_thresh,
        config.yolo_v4_loss.bg_iou_thresh,
        config.yolo_v4_loss.bbox_loss_type,
    )
    return criterion
