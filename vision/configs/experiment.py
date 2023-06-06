import os.path
from typing import Tuple, Optional

from vision.configs import register_experiment_config
from vision.configs.classification import ClassificationModel
from vision.configs.detection import DetectionModel
from vision.configs.dataset import Dataset, Augmentation, AugPolicy
from vision.configs.loss import Loss, CrossEntropyLoss, YOLOv4Loss
from vision.configs.lr_scheduler import LRScheduler
from vision.configs.optimizer import Optimizer, Adam
from vision.configs.task import Trainer, ClassificationTask, DetectionTask


DATASET_BASE_DIR = os.path.join("..", "datasets")

DOG_VS_CAT_DIR = "dog_vs_cat"
DOG_VS_CAT_BASE_TRAIN_LABEL = "train.json"
DOG_VS_CAT_BASE_VAL_LABEL = "val.json"
DOG_VS_CAT_BASE_IMAGE_DIR_NAME = "images"

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


@register_experiment_config("dog_vs_cat_classification_resnet")
def dog_vs_cat_classification_resnet():
    epochs: int = 100
    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 256
    num_workers: int = 4
    learning_rate: float = 1e-4

    exp_config = Trainer(
        type="classification",
        logger="tensorboard",
        classification=ClassificationTask(
            classification_model=ClassificationModel(),
            optimizer=Optimizer(type="adam", lr=learning_rate, adam=Adam()),
            lr_scheduler=LRScheduler(
                type="one_cycle_lr",
            ),
            loss=Loss(
                type="cross_entropy_loss",
                cross_entropy_loss=CrossEntropyLoss(label_smoothing=0.1),
            ),
        ),
        epochs=epochs,
        train_data=Dataset(
            type="classification",
            image_dir=os.path.join(
                DATASET_BASE_DIR, DOG_VS_CAT_DIR, DOG_VS_CAT_BASE_IMAGE_DIR_NAME
            ),
            label_path=os.path.join(
                DATASET_BASE_DIR, DOG_VS_CAT_DIR, DOG_VS_CAT_BASE_TRAIN_LABEL
            ),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            augmentation=Augmentation(aug_list=AugPolicy.simple_aug(image_size)),
        ),
        val_data=Dataset(
            type="classification",
            image_dir=os.path.join(
                DATASET_BASE_DIR, DOG_VS_CAT_DIR, DOG_VS_CAT_BASE_IMAGE_DIR_NAME
            ),
            label_path=os.path.join(
                DATASET_BASE_DIR, DOG_VS_CAT_DIR, DOG_VS_CAT_BASE_VAL_LABEL
            ),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            augmentation=Augmentation(aug_list=AugPolicy.val_aug(image_size)),
        ),
    )

    return exp_config


@register_experiment_config("coco_yolo")
def coco_yolo():
    epochs: int = 200
    image_size: Tuple[int, int, int] = (3, 640, 640)
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 5e-4
    num_classes: Optional[int] = None

    exp_config = Trainer(
        type="detection",
        logger="tensorboard",
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
            image_dir=os.path.join(DATASET_BASE_DIR, COCO_DIR, COCO_TRAIN_IMAGE_DIR),
            label_path=os.path.join(DATASET_BASE_DIR, COCO_DIR, COCO_TRAIN_LABEL),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            augmentation=Augmentation(aug_list=AugPolicy.simple_aug(image_size)),
        ),
        val_data=Dataset(
            type="coco",
            image_dir=os.path.join(DATASET_BASE_DIR, COCO_DIR, COCO_VAL_IMAGE_DIR),
            label_path=os.path.join(DATASET_BASE_DIR, COCO_DIR, COCO_VAL_LABEL),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            augmentation=Augmentation(aug_list=AugPolicy.val_aug(image_size)),
        ),
    )

    return exp_config


@register_experiment_config("voc_yolo")
def voc_yolo():
    epochs: int = 200
    image_size: Tuple[int, int, int] = (3, 640, 640)
    batch_size: int = 32
    num_workers: int = 16
    learning_rate: float = 5e-4
    num_classes: Optional[int] = None

    exp_config = Trainer(
        type="detection",
        logger="tensorboard",
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
            image_dir=os.path.join(DATASET_BASE_DIR, VOC_DIR, VOC_TRAIN_IMAGE_DIR),
            label_path=os.path.join(DATASET_BASE_DIR, VOC_DIR, VOC_TRAIN_LABEL),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            augmentation=Augmentation(aug_list=AugPolicy.simple_aug(image_size)),
        ),
        val_data=Dataset(
            type="coco",
            image_dir=os.path.join(DATASET_BASE_DIR, VOC_DIR, VOC_TRAIN_IMAGE_DIR),
            label_path=os.path.join(DATASET_BASE_DIR, VOC_DIR, VOC_TRAIN_LABEL),
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            augmentation=Augmentation(aug_list=AugPolicy.val_aug(image_size)),
        ),
    )

    return exp_config
