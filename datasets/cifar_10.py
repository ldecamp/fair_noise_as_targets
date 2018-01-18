#!/usr/bin/python
import numpy as np
from typing import List, Tuple

from torchvision.datasets import CIFAR10

from core.dataloader import DataLoader

from datasets.utils import generate_random_targets
from datasets.interface import DynamicTargetDataset


class NatCIFAR10(CIFAR10, DynamicTargetDataset):

    def __init__(self,
                 train: bool=True,
                 z_dims: int=50,
                 **kwargs) -> None:
        """
        Override Cifar 10 Dataset to also stream randomly generated encoder targets.

        :param train: whether to stream train or test
        :param stream_nat: whether to stream cifar class labels or NAT targets
        :param z_dims: encoder output features space dimensionality
        :param kwargs: any other dataset args.
        """
        super().__init__(train=train, **kwargs)

        # randomly create targets.
        if train:
            self.train_nat = generate_random_targets(len(self.train_data), z_dims)
        else:
            self.test_nat = np.empty(shape=(len(self.test_data), z_dims))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = super().__getitem__(index)

        if not self.train:
            return x, y, self.test_nat[index, :]
        else:
            return x, y, self.train_nat[index, :]

    def update_targets(self, indexes: List[int], new_targets: np.ndarray):
        """
        Helper method that update the assigned feature representation
        Used after cost minimisation every few epoch.
        :param indexes:
        :param new_targets:
        :return:
        """
        if self.train:
            self.train_nat[indexes, :] = new_targets
        else:
            self.test_nat[indexes, :] = new_targets


if __name__ == '__main__':
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('L')),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Does [0; 1] Standardisation
    ])

    trainset = NatCIFAR10(root='./data', train=True,
                          download=True, transform=transform)

    dl_train = DataLoader(trainset,
                          batch_size=2,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=False)

    # Test streaming and override targets.
    for i in range(2):
        if i == 0:
            for idx, x, y, nat in dl_train:
                trainset.update_targets(idx, np.zeros(shape=(2, 50)))
        else:
            for idx, x, y, nat in dl_train:
                assert np.array_equal(nat.numpy(), np.zeros(nat.shape, dtype=np.float32))
