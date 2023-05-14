import numpy as np
import os.path
from typing import Dict

from torch import Tensor
import albumentations as A
from pycocotools.coco import COCO
import cv2
from torch.utils.data import DataLoader

from vision.configs import dataset as dataset_config
from vision.dataloader import register_dataloader
from vision.dataloader.core import BaseDataset
from vision.dataloader.utils import parse_augmentation


class COCODataset(BaseDataset):
    def __init__(self, config: dataset_config.Dataset) -> None:
        super().__init__()
        self.coco = COCO(config.label_path)
        self.image_dir_path = config.image_dir
        self.max_objects = config.max_objects
        self.image_ids = self.coco.getImgIds()

        self.transforms: A.Compose = parse_augmentation(
            aug_config=config.augmentation, bbox=True
        )

    def __getitem__(self, i) -> Dict[str, Tensor]:
        result = {}
        image_id = self.image_ids[i]
        image_info = self.coco.loadImgs(image_id)[0]
        annotation_id = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(annotation_id)

        boxes = []
        categories = []

        for annotation in annotations:
            if annotation["iscrowd"]:
                continue
            boxes += [annotation["bbox"]]
            categories += [annotation["category_id"]]

        boxes = boxes[: self.max_objects]
        categories = categories[: self.max_objects]

        image = cv2.imread(os.path.join(self.image_dir_path, image_info["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        result["image"] = image
        result["bboxes"] = boxes
        result["categories"] = categories

        result = self.transforms(**result)

        result["bboxes"] += [[-1, -1, -1, -1]] * (
            self.max_objects - len(result["bboxes"])
        )
        result["categories"] += [-1] * (self.max_objects - len(result["categories"]))

        result["bboxes"] = np.array(result["bboxes"])
        result["categories"] = np.array(result["categories"])

        return result

    def __len__(self) -> int:
        return len(self.coco.getImgIds())

    def get_num_classes(self) -> int:
        return len(self.coco.getCatIds())


@register_dataloader("coco")
def coco_dataloader(config: dataset_config.Dataset):
    assert config.type == "coco"

    dataset = COCODataset(config)
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
