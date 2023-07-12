from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def get_num_classes(self) -> int:
        pass

    @abstractmethod
    def __len__(self):
        pass
