import dataclasses
from typing import Sequence, Tuple, Optional, Dict, Any

import cv2


@dataclasses.dataclass
class AugPolicy:
    @staticmethod
    def color_aug(
        image_size: Tuple[int, int, int]
    ) -> Sequence[Tuple[str, Dict[str, Any]]]:
        return (
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
        )

    @staticmethod
    def simple_aug(
        image_size: Tuple[int, int, int]
    ) -> Sequence[Tuple[str, Dict[str, Any]]]:
        return (
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
        )

    @staticmethod
    def val_aug(
        image_size: Tuple[int, int, int]
    ) -> Sequence[Tuple[str, Dict[str, Any]]]:
        return (
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
        )

    @staticmethod
    def geometric_aug(
        image_size: Tuple[int, int, int]
    ) -> Sequence[Tuple[str, Dict[str, Any]]]:
        return (
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
        )


@dataclasses.dataclass
class Augmentation:
    aug_list: Sequence[Tuple[str, Dict[str, Any]]] = AugPolicy.val_aug((3, 256, 256))


@dataclasses.dataclass
class Dataset:
    type: Optional[str] = None
    image_dir: Optional[str] = None
    label_path: Optional[str] = None
    is_train: bool = False

    image_size: Tuple[int, int, int] = (3, 256, 256)
    batch_size: int = 256
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""

    augmentation: Augmentation = Augmentation()
