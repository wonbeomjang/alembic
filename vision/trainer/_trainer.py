from abc import ABCMeta, abstractmethod

from torch.utils.data import DataLoader


class BasicTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        pass

    @abstractmethod
    def test(self, test_dataloader: DataLoader):
        pass
