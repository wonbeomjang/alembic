import os.path
from typing import Tuple

import cv2

from vision.configs import register_experiment_config
from vision.configs.classification import ClassificationModel
from vision.configs.dataset import Dataset, Augmentation
from vision.configs.loss import Loss, CrossEntropyLoss
from vision.configs.optimizer import Optimizer, Adam
from vision.configs.trainer import Trainer, ClassificationTrainer


DOG_VS_CAT_BASE_DIR = os.path.join("..", "datasets", "dog_vs_cat")
DOG_VS_CAT_BASE_TRAIN_LABEL = "train.json"
DOG_VS_CAT_BASE_VAL_LABEL = "val.json"
DOG_VS_CAT_BASE_IMAGE_DIR_NAME = "images"


NEUROCLE_BASE_DIR = os.path.join("..", "datasets", "cla")
NEUROCLE_LABEL = "hanlim_label.json"
NEUROCLE_IMAGE_DIR = "images"


@register_experiment_config("dog_vs_cat_classification_resnet")
def dog_vs_cat_classification_resnet():
    epochs: int = 100
    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 32
    num_workers: int = 4

    exp_config = Trainer(
        type="classification",
        classification=ClassificationTrainer(
            classification_model=ClassificationModel(num_classes=2),
            optimizer=Optimizer(type="adam", adam=Adam(lr=1e-3)),
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
                    ("LongestMaxSize", {"max_size": image_size}),
                    (
                        "PadIfNeeded",
                        {
                            "min_height": image_size,
                            "min_width": image_size,
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
                    ("LongestMaxSize", {"max_size": image_size}),
                    (
                        "PadIfNeeded",
                        {
                            "min_height": image_size,
                            "min_width": image_size,
                            "border_mode": cv2.BORDER_CONSTANT,
                        },
                    ),
                    ("Normalize", {}),
                ]
            ),
        ),
    )

    return exp_config
