from typing import Dict, Optional, Tuple, Union, List

import torch
from torch import nn, Tensor
from torchvision.models.detection import _utils as det_utils
from torchvision import ops as det_ops

from vision.configs import detection as yolo_cfg

from vision.configs.base_config import ModelConfig
from vision.modeling import register_model
from vision.modeling.backbones import get_backbone
from vision.modeling.head import get_head
from vision.modeling.necks import get_neck
from vision.utils.anchor import AnchorGenerator


class YOLO(nn.Module):
    def __init__(
        self,
        model_config: yolo_cfg.YOLO,
        anchor_generator: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = get_backbone(model_config.backbone)

        with torch.no_grad():
            dummy_input = torch.randn((1, 3, 224, 224))
            output = self.backbone(dummy_input)

        in_channels = {}
        for k in output.keys():
            in_channels[k] = output[k].size(1)

        model_config.neck.fpn.in_channels = in_channels
        model_config.head.yolo._min_level = model_config.neck.fpn.min_level
        model_config.head.yolo._max_level = model_config.neck.fpn.max_level

        self.neck = get_neck(model_config.neck)
        self.head = get_head(model_config.head)

        if anchor_generator is None:
            anchor_generator = AnchorGenerator()
        self.anchor_generator = anchor_generator
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.anchor: Optional[Dict[str, Tensor]] = None
        self.iou_threshold = model_config.iou_threshold
        self.is_train = True
        self.num_images = -1

    def forward(
        self, x: Tuple[Tensor]
    ) -> Union[Dict[str, Dict[str, Tensor]], Tuple[Tensor, List[Dict[str, Tensor]]]]:
        x = torch.stack(x)

        feature = self.backbone(x)
        feature = self.neck(feature)

        if self.num_images != x.shape[0]:
            self.anchor = self.anchor_generator(x, feature)

        x = self.head(feature)
        if self.training:
            return x

        pred_bboxes = x["boxes"]
        pred_labels = x["labels"]

        result = []

        for dt_label, dt_bbox, anchor in zip(pred_labels, pred_bboxes, self.anchor):
            dt_score, dt_class = torch.max(dt_label, dim=1)
            foreground_index = dt_class != 0

            dt_class = dt_class[foreground_index]
            dt_score = dt_score[foreground_index]
            dt_bbox = dt_bbox[foreground_index]
            anchor = anchor[foreground_index]

            dt_bbox = self.box_coder.decode_single(dt_bbox, anchor)
            nms_index = det_ops.nms(dt_bbox, dt_score, self.iou_threshold)

            dt_class = dt_class[nms_index]
            dt_score = dt_score[nms_index]
            dt_bbox = dt_bbox[nms_index]

            result += [
                {"category_ids": dt_class, "bboxes": dt_bbox, "scores": dt_score}
            ]

        return x, result


@register_model("yolo")
def yolo(model_cfg: ModelConfig):
    assert isinstance(model_cfg, yolo_cfg.DetectionModel)
    if model_cfg.yolo.head.yolo._min_level != model_cfg.yolo.neck.fpn.min_level:
        model_cfg.yolo.head.yolo._min_level = model_cfg.yolo.neck.fpn.min_level
    if model_cfg.yolo.head.yolo._max_level != model_cfg.yolo.neck.fpn.max_level:
        model_cfg.yolo.head.yolo._max_level = model_cfg.yolo.neck.fpn.max_level
    if model_cfg.yolo.head._num_classes != model_cfg.num_classes:
        model_cfg.yolo.head._num_classes = model_cfg.num_classes

    model = YOLO(model_cfg.yolo)

    return model
