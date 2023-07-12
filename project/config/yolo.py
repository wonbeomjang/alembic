import os
from typing import Optional, Tuple

from vision.configs import register_experiment_config
from vision.configs.dataset import Dataset, Augmentation, AugPolicy
from vision.configs.detection import DetectionModel
from vision.configs.experiment import ExperimentConfig
from vision.configs.loss import Loss, YOLOv4Loss
from vision.configs.lr_scheduler import LRScheduler
from vision.configs.optimizer import Optimizer, Adam
from vision.configs.task import Task, DetectionTask


DATASET_BASE_DIR = os.path.join("..", "datasets")

COCO_DIR = "coco"
COCO_TRAIN_LABEL = os.path.join("annotations", "instances_train2017.json")
COCO_TRAIN_IMAGE_DIR = os.path.join("images", "train2017")
COCO_VAL_LABEL = os.path.join("annotations", "instances_val2017.json")
COCO_VAL_IMAGE_DIR = os.path.join("images", "val2017")

VOC_DIR = "voc"
VOC_TRAIN_LABEL = os.path.join("VOC2012_train_val", "coco_label.json")
VOC_TRAIN_IMAGE_DIR = os.path.join("VOC2012_train_val", "JPEGImages")
VOC_VAL_LABEL = os.path.join("VOC2012_test", "coco_label.json")
VOC_VAL_IMAGE_DIR = os.path.join("VOC2012_test", "JPEGImages")


@register_experiment_config("coco_yolo")
def coco_yolo():
    epochs: int = 200
    image_size: Tuple[int, int, int] = (3, 640, 640)
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 5e-4
    num_classes: Optional[int] = None
    log_dir: str = os.path.join(".", "logs")

    train_image_dir: str = os.path.join(
        DATASET_BASE_DIR, COCO_DIR, COCO_TRAIN_IMAGE_DIR
    )
    train_label_path: str = os.path.join(DATASET_BASE_DIR, COCO_DIR, COCO_TRAIN_LABEL)
    val_image_dir: str = os.path.join(DATASET_BASE_DIR, COCO_DIR, COCO_VAL_IMAGE_DIR)
    val_label_path: str = os.path.join(DATASET_BASE_DIR, COCO_DIR, COCO_VAL_LABEL)

    exp_config = base_yolo_config(
        batch_size,
        epochs,
        image_size,
        learning_rate,
        log_dir,
        num_classes,
        num_workers,
        train_image_dir,
        train_label_path,
        val_image_dir,
        val_label_path,
    )

    return exp_config


@register_experiment_config("voc_yolo")
def voc_yolo():
    epochs: int = 200
    image_size: Tuple[int, int, int] = (3, 640, 640)
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 5e-4
    num_classes: Optional[int] = None
    log_dir: str = os.path.join(".", "logs")

    train_image_dir: str = os.path.join(DATASET_BASE_DIR, VOC_DIR, VOC_TRAIN_IMAGE_DIR)
    train_label_path: str = os.path.join(DATASET_BASE_DIR, VOC_DIR, VOC_TRAIN_LABEL)
    val_image_dir: str = os.path.join(DATASET_BASE_DIR, VOC_DIR, VOC_VAL_IMAGE_DIR)
    val_label_path: str = os.path.join(DATASET_BASE_DIR, VOC_DIR, VOC_VAL_LABEL)

    exp_config = base_yolo_config(
        batch_size,
        epochs,
        image_size,
        learning_rate,
        log_dir,
        num_classes,
        num_workers,
        train_image_dir,
        train_label_path,
        val_image_dir,
        val_label_path,
    )

    return exp_config


def base_yolo_config(
    batch_size: int,
    epochs: int,
    image_size: Tuple[int, int, int],
    learning_rate: float,
    log_dir,
    num_classes: int,
    num_workers: int,
    train_image_dir: str,
    train_label_path: str,
    val_image_dir: str,
    val_label_path: str,
):
    exp_config = ExperimentConfig(
        task=Task(
            type="detection",
            detection=DetectionTask(
                model=DetectionModel(
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
        ),
        train_data=Dataset(
            type="coco",
            image_dir=train_image_dir,
            label_path=train_label_path,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            augmentation=Augmentation(aug_list=AugPolicy.simple_aug(image_size)),
        ),
        val_data=Dataset(
            type="coco",
            image_dir=val_image_dir,
            label_path=val_label_path,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            augmentation=Augmentation(aug_list=AugPolicy.val_aug(image_size)),
        ),
        epochs=epochs,
        logger="tensorboard",
        log_dir=log_dir,
    )
    return exp_config
