from typing import Sequence, List

import albumentations as A
from albumentations import BoxType
from albumentations.core.bbox_utils import (
    BboxProcessor,
    convert_bboxes_to_albumentations,
)
from albumentations.pytorch import ToTensorV2

from vision.configs import dataset as dataset_config

ALBUMENTATION_CHECK_VALIDITY = False


if ALBUMENTATION_CHECK_VALIDITY:

    def _convert_to_albumentations(
        self, data: Sequence[BoxType], rows: int, cols: int
    ) -> List[BoxType]:
        return convert_bboxes_to_albumentations(
            data, self.params.format, rows, cols, check_validity=False
        )

    BboxProcessor.convert_to_albumentations = _convert_to_albumentations


def parse_augmentation(
    aug_config: dataset_config.Augmentation, bbox=False
) -> A.Compose:
    compose_list = []

    for aug, kwargs in aug_config.aug_list:
        compose_list += [getattr(A, aug)(**kwargs)]

    compose_list += [ToTensorV2()]
    if bbox:
        transform = A.Compose(
            compose_list,
            bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
        )
    else:
        transform = A.Compose(compose_list)

    return transform
