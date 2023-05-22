from abc import ABCMeta, abstractmethod

from torch.utils.data import DataLoader


class BasicTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train_and_eval(self):
        pass

    @abstractmethod
    def test(self, test_dataloader: DataLoader):
        pass

    @abstractmethod
    def predict(self, dataloader: DataLoader):
        pass
