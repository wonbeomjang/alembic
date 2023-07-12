import os
from typing import Tuple

from vision.configs import register_experiment_config
from vision.configs.classification import ClassificationModel
from vision.configs.dataset import Dataset, Augmentation, AugPolicy
from vision.configs.experiment import ExperimentConfig
from vision.configs.loss import Loss, CrossEntropyLoss
from vision.configs.lr_scheduler import LRScheduler
from vision.configs.optimizer import Optimizer, Adam
from vision.configs.task import Task, ClassificationTask

DATASET_BASE_DIR = os.path.join("..", "datasets")

DOG_VS_CAT_DIR = "dog_vs_cat"
DOG_VS_CAT_BASE_TRAIN_LABEL = "train.json"
DOG_VS_CAT_BASE_VAL_LABEL = "val.json"
DOG_VS_CAT_BASE_IMAGE_DIR_NAME = "images"


@register_experiment_config("dog_vs_cat_classification_resnet")
def dog_vs_cat_classification_resnet():
    epochs: int = 100
    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 256
    num_workers: int = 4
    learning_rate: float = 1e-4

    exp_config = ExperimentConfig(
        task=Task(
            type="classification",
            classification=ClassificationTask(
                model=ClassificationModel(),
                optimizer=Optimizer(type="adam", lr=learning_rate, adam=Adam()),
                lr_scheduler=LRScheduler(
                    type="one_cycle_lr",
                ),
                loss=Loss(
                    type="cross_entropy_loss",
                    cross_entropy_loss=CrossEntropyLoss(label_smoothing=0.1),
                ),
            ),
        ),
        logger="tensorboard",
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
