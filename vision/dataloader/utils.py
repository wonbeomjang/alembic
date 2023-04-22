import albumentations as A
from albumentations.pytorch import ToTensorV2

from vision.configs import dataset as dataset_config


def parse_augmentation(aug_config: dataset_config.Augmentation) -> A.Compose:
    compose_list = []

    for aug, kwargs in aug_config.aug_list:
        compose_list += [getattr(A, aug)(**kwargs)]

    compose_list += [ToTensorV2()]
    transform = A.Compose(compose_list)

    return transform
