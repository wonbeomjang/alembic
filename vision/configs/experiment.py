import os.path
from typing import Tuple

import cv2

from vision.configs import register_experiment_config, backbones
from vision.configs.classification import ClassificationModel
from vision.configs.dataset import Dataset, Augmentation
from vision.configs.loss import Loss, CrossEntropyLoss
from vision.configs.lr_scheduler import LRScheduler, OneCycleLR
from vision.configs.optimizer import Optimizer, Adam
from vision.configs.trainer import Trainer, ClassificationTrainer


DOG_VS_CAT_BASE_DIR = os.path.join("..", "datasets", "dog_vs_cat")
DOG_VS_CAT_BASE_TRAIN_LABEL = "train.json"
DOG_VS_CAT_BASE_VAL_LABEL = "val.json"
DOG_VS_CAT_BASE_IMAGE_DIR_NAME = "images"


NEUROCLE_BASE_DIR = "/app/input/dataset/classification-pytorch"
# NEUROCLE_BASE_DIR = os.path.join("..", "datasets", "cla")
NEUROCLE_LABEL = "hanlim_label.json"
NEUROCLE_IMAGE_DIR = "images"


@register_experiment_config("dog_vs_cat_classification_resnet")
def dog_vs_cat_classification_resnet():
    epochs: int = 100
    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 256
    num_workers: int = 4
    learning_rate: float = 1e-4

    exp_config = Trainer(
        type="classification",
        classification=ClassificationTrainer(
            classification_model=ClassificationModel(),
            optimizer=Optimizer(type="adam", lr=learning_rate, adam=Adam()),
            loss=Loss(
                type="cross_entropy_loss",
                cross_entropy_loss=CrossEntropyLoss(label_smoothing=0.1),
            ),
        ),
        epochs=epochs,
        train_data=Dataset(
            type="classification",
            image_dir=os.path.join(DOG_VS_CAT_BASE_DIR, DOG_VS_CAT_BASE_IMAGE_DIR_NAME),
            label_path=os.path.join(DOG_VS_CAT_BASE_DIR, DOG_VS_CAT_BASE_TRAIN_LABEL),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            augmentation=Augmentation(
                aug_list=[
                    ("LongestMaxSize", {"max_size": max(image_size[1], image_size[2])}),
                    (
                        "PadIfNeeded",
                        {
                            "min_height": image_size[1],
                            "min_width": image_size[2],
                            "border_mode": cv2.BORDER_CONSTANT,
                        },
                    ),
                    ("HorizontalFlip", {}),
                    ("Normalize", {}),
                ]
            ),
        ),
        val_data=Dataset(
            type="classification",
            image_dir=os.path.join(DOG_VS_CAT_BASE_DIR, DOG_VS_CAT_BASE_IMAGE_DIR_NAME),
            label_path=os.path.join(DOG_VS_CAT_BASE_DIR, DOG_VS_CAT_BASE_VAL_LABEL),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            augmentation=Augmentation(
                aug_list=[
                    ("LongestMaxSize", {"max_size": max(image_size[1], image_size[2])}),
                    (
                        "PadIfNeeded",
                        {
                            "min_height": image_size[1],
                            "min_width": image_size[2],
                            "border_mode": cv2.BORDER_CONSTANT,
                        },
                    ),
                    ("Normalize", {}),
                ]
            ),
        ),
    )

    return exp_config


@register_experiment_config("neurocle_cla_resnet")
def neurocle_cla_resnet():
    epochs: int = 300
    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 512
    num_workers: int = 4
    learning_rate: float = 1e-3

    exp_config = Trainer(
        type="classification",
        logger="tensorboard",
        save_best_model=True,
        epochs=epochs,
        classification=ClassificationTrainer(
            classification_model=ClassificationModel(),
            optimizer=Optimizer(type="adam", lr=learning_rate, adam=Adam()),
            lr_scheduler=LRScheduler(
                type="one_cycle_lr",
                one_cycle_lr=OneCycleLR(),
            ),
            loss=Loss(
                type="cross_entropy_loss",
                cross_entropy_loss=CrossEntropyLoss(),
            ),
        ),
        train_data=Dataset(
            type="neurocle_classification",
            image_dir=os.path.join(NEUROCLE_BASE_DIR, NEUROCLE_IMAGE_DIR),
            label_path=os.path.join(NEUROCLE_BASE_DIR, NEUROCLE_LABEL),
            is_train=True,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            augmentation=Augmentation(
                aug_list=[
                    ("LongestMaxSize", {"max_size": max(image_size)}),
                    (
                        "PadIfNeeded",
                        {
                            "min_height": image_size[1],
                            "min_width": image_size[2],
                            "border_mode": cv2.BORDER_CONSTANT,
                        },
                    ),
                    ("HorizontalFlip", {}),
                    ("VerticalFlip", {}),
                    ("GaussianBlur", {"p": 0.2}),
                    ("RandomBrightnessContrast", {"p": 0.2}),
                    ("RandomGamma", {"p": 0.2}),
                    ("Rotate", {"limit": 180}),
                    ("Normalize", {}),
                ]
            ),
        ),
        val_data=Dataset(
            type="neurocle_classification",
            image_dir=os.path.join(NEUROCLE_BASE_DIR, NEUROCLE_IMAGE_DIR),
            label_path=os.path.join(NEUROCLE_BASE_DIR, NEUROCLE_LABEL),
            is_train=False,
            augmentation=Augmentation(
                aug_list=[
                    ("LongestMaxSize", {"max_size": max(image_size)}),
                    (
                        "PadIfNeeded",
                        {
                            "min_height": image_size[1],
                            "min_width": image_size[2],
                            "border_mode": cv2.BORDER_CONSTANT,
                        },
                    ),
                    ("Normalize", {}),
                ]
            ),
        ),
    )

    return exp_config


@register_experiment_config("neurocle_cla_led")
def neurocle_cla_led():
    epochs: int = 400
    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 256
    num_workers: int = 16
    learning_rate: float = 5e-4

    base_dir = os.path.join("..", "datasets", "neurocle_cla")
    log_dir = "led"

    # base_dir = "/app/input/dataset/classification"
    # log_dir = "/app/outputs"
    image_dir = os.path.join(base_dir, "led", "image")
    label_path = os.path.join(base_dir, "led", "label.json")

    return neurocle_cla_geometric_aug_base(
        batch_size,
        epochs,
        image_dir,
        image_size,
        label_path,
        learning_rate,
        log_dir,
        num_workers,
    )


def neurocle_cla_geometric_aug_base(
    batch_size: int,
    epochs: int,
    image_dir: str,
    image_size: Tuple[int, int, int],
    label_path: str,
    learning_rate: float,
    log_dir: str,
    num_workers: int,
):
    exp_config = Trainer(
        type="classification",
        logger="tensorboard",
        log_dir=log_dir,
        save_best_model=True,
        classification=ClassificationTrainer(
            classification_model=ClassificationModel(
                backbone=backbones.Backbone(
                    type="alembic_mobilenet",
                    pretrained=False,
                )
            ),
            optimizer=Optimizer(type="adam", lr=learning_rate, adam=Adam()),
            loss=Loss(
                type="cross_entropy_loss",
                cross_entropy_loss=CrossEntropyLoss(),
            ),
        ),
        epochs=epochs,
        train_data=Dataset(
            type="neurocle_classification",
            image_dir=image_dir,
            label_path=label_path,
            is_train=True,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            augmentation=Augmentation(
                aug_list=[
                    ("LongestMaxSize", {"max_size": max(image_size)}),
                    (
                        "PadIfNeeded",
                        {
                            "min_height": image_size[1],
                            "min_width": image_size[2],
                            "border_mode": cv2.BORDER_CONSTANT,
                        },
                    ),
                    ("Affine", {"translate_percent": 0.1}),
                    ("HorizontalFlip", {}),
                    ("VerticalFlip", {}),
                    ("Normalize", {}),
                ]
            ),
        ),
        val_data=Dataset(
            type="neurocle_classification",
            image_dir=image_dir,
            label_path=label_path,
            is_train=False,
            augmentation=Augmentation(
                aug_list=[
                    ("LongestMaxSize", {"max_size": max(image_size)}),
                    (
                        "PadIfNeeded",
                        {
                            "min_height": image_size[1],
                            "min_width": image_size[2],
                            "border_mode": cv2.BORDER_CONSTANT,
                        },
                    ),
                    ("Normalize", {}),
                ]
            ),
        ),
    )
    return exp_config
