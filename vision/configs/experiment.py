import os.path
from typing import Tuple, Optional

import cv2

from vision.configs import register_experiment_config
from vision.configs.classification import ClassificationModel
from vision.configs.dataset import Dataset, Augmentation
from vision.configs.detection import DetectionModel
from vision.configs.loss import Loss, CrossEntropyLoss, YOLOv4Loss
from vision.configs.lr_scheduler import LRScheduler
from vision.configs.optimizer import Optimizer, Adam
from vision.configs.task import Trainer, ClassificationTask, DetectionTask

DOG_VS_CAT_BASE_DIR = os.path.join("..", "datasets", "dog_vs_cat")
DOG_VS_CAT_BASE_TRAIN_LABEL = "train.json"
DOG_VS_CAT_BASE_VAL_LABEL = "val.json"
DOG_VS_CAT_BASE_IMAGE_DIR_NAME = "images"

COCO_BASE_DIR = os.path.join("..", "datasets", "coco")
COCO_TRAIN_LABEL = os.path.join("annotations", "instances_train2017.json")
COCO_TRAIN_IMAGE_DIR = os.path.join("images", "train2017")
COCO_VAL_LABEL = os.path.join("annotations", "instances_val2017.json")
COCO_VAL_IMAGE_DIR = os.path.join("images", "val2017")


@register_experiment_config("dog_vs_cat_classification_resnet")
def dog_vs_cat_classification_resnet():
    epochs: int = 100
    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-3

    exp_config = Trainer(
        type="classification",
        classification=ClassificationTask(
            classification_model=ClassificationModel(num_classes=2),
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
            type="classification",
            image_dir=os.path.join(DOG_VS_CAT_BASE_DIR, DOG_VS_CAT_BASE_IMAGE_DIR_NAME),
            label_path=os.path.join(DOG_VS_CAT_BASE_DIR, DOG_VS_CAT_BASE_VAL_LABEL),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
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
                    ("Normalize", {}),
                ]
            ),
        ),
    )

    return exp_config


@register_experiment_config("coco_yolo")
def coco_yolo():
    epochs: int = 100
    image_size: Tuple[int, int, int] = (3, 512, 512)
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 1e-4
    num_classes: Optional[int] = None

    exp_config = Trainer(
        type="detection",
        detection=DetectionTask(
            detection_model=DetectionModel(
                type="yolo",
                num_classes=num_classes,
            ),
            optimizer=Optimizer(type="adam", lr=learning_rate, adam=Adam()),
            lr_scheduler=LRScheduler(type="one_cycle_lr"),
            loss=Loss(
                type="yolo_v4_loss",
                yolo_v4_loss=YOLOv4Loss(
                    bbox_loss_type="smooth_l1",
                ),
            ),
        ),
        epochs=epochs,
        train_data=Dataset(
            type="coco",
            image_dir=os.path.join(COCO_BASE_DIR, COCO_TRAIN_IMAGE_DIR),
            label_path=os.path.join(COCO_BASE_DIR, COCO_TRAIN_LABEL),
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
                    ("Normalize", {}),
                ]
            ),
        ),
        val_data=Dataset(
            type="coco",
            image_dir=os.path.join(COCO_BASE_DIR, COCO_VAL_IMAGE_DIR),
            label_path=os.path.join(COCO_BASE_DIR, COCO_VAL_LABEL),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
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
                    ("Normalize", {}),
                ]
            ),
        ),
    )

    return exp_config
