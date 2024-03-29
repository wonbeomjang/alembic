import json
import numpy as np
import os.path
from typing import Dict, Tuple

from torch import Tensor
from torch.utils.data import DataLoader
import albumentations as A
import cv2

from vision.configs import dataset as dataset_config
from vision.dataloader import register_dataloader
from vision.dataloader.core import BaseDataset
from vision.dataloader.utils import parse_augmentation


class ImageClassificationDataset(BaseDataset):
    def __init__(self, config: dataset_config.Dataset) -> None:
        super().__init__()
        with open(config.label_path) as f:
            labels: Dict = json.load(f)

        categories = labels["categories"]

        self.labels = labels["labels"]
        self.category_to_index: Dict[str, int] = {
            c: i for i, c in enumerate(categories)
        }
        self.image_dir_path = config.image_dir

        self.transforms: A.Compose = parse_augmentation(aug_config=config.augmentation)

    def __getitem__(self, i) -> Tuple[Tensor, int]:
        image_id = self.labels[i]["image_id"]
        category = self.labels[i]["categories"][0]
        label = self.category_to_index[category]

        image = cv2.imread(os.path.join(self.image_dir_path, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = self.transforms(image=image)["image"]

        return image, label

    def __len__(self) -> int:
        return len(self.labels)

    def get_num_classes(self) -> int:
        return len(self.category_to_index)


@register_dataloader("classification")
def classification_dataloader(config: dataset_config.Dataset):
    assert config.type == "classification"

    dataset = ImageClassificationDataset(config)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        # pin_memory=config.pin_memory,
        # drop_last=config.drop_last,
        # timeout=config.timeout,
        # prefetch_factor=config.prefetch_factor,
        # persistent_workers=config.persistent_workers,
        # pin_memory_device=config.pin_memory_device,
    )

    return dataloader
