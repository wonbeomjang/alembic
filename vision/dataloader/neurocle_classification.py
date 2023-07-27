import json
import os
from collections import defaultdict
from typing import List, Any, Set, Tuple, Dict

import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import albumentations as A

from vision.configs import dataset as dataset_config
from vision.dataloader import register_dataloader
from vision.dataloader.utils import parse_augmentation


def imread_utf8(path: str):
    stream = open(path.encode("utf-8"), "rb")
    image = np.asarray(bytearray(stream.read()), dtype=np.uint8)

    return cv2.imdecode(image, cv2.IMREAD_UNCHANGED)


class NeurocleClassificationDataset(Dataset):
    def __init__(self, config: dataset_config.Dataset):
        label_set: str = "train" if config.is_train else "test"

        with open(config.label_path, encoding="utf-8") as f:
            self.labels: List[Dict[str, Any]] = json.load(f)["data"]

        label_info = defaultdict(int)

        self.class_index: Set[str] = set(
            [d["classLabel"] for d in self.labels if d["classLabel"]]
        )
        self.class_index: Dict[str, int] = {
            d: i for i, d in enumerate(self.class_index)
        }

        new_label: List[Dict[str, Any]] = []

        for d in self.labels:
            if d["set"] == label_set and d["classLabel"]:
                new_label += [d]
                label_info[d["classLabel"]] += 1

        self.labels = new_label
        self.image_dir_path = config.image_dir
        self.transforms: A.Compose = parse_augmentation(aug_config=config.augmentation)

    def __getitem__(self, i) -> Tuple[Tensor, int]:
        label = self.labels[i]
        image = imread_utf8(os.path.join(self.image_dir_path, label["fileName"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = self.transforms(image=image)["image"]

        return image, self.class_index[label["classLabel"]]

    def __len__(self) -> int:
        return len(self.labels)

    def get_num_classes(self) -> int:
        return len(self.class_index)


@register_dataloader("neurocle_classification")
def neurocle_classification(config: dataset_config.Dataset):
    assert config.type == "neurocle_classification"

    dataset = NeurocleClassificationDataset(config)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        timeout=config.timeout,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory_device=config.pin_memory_device,
    )

    return dataloader
