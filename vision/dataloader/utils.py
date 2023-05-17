import albumentations as A
from albumentations.pytorch import ToTensorV2

from vision.configs import dataset as dataset_config


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
