#!/usr/bin/python
import numpy as np
from typing import Callable, List, Tuple

# import torch.multiprocessing

import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler, BatchSampler, RandomSampler, SequentialSampler
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import pin_memory_batch, default_collate

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


class NatDataLoaderIter(object):
    """
    Defines a data loader iterator that returns minibatch as well as the items index in the dataframe.

    Doesn't support multi processing yet. Need to work on target sync first.
    TODO: Add multi processing support.
    """

    def __init__(self, loader) -> None:
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.pin_memory = loader.pin_memory

        self.sample_iter = iter(self.batch_sampler)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __next__(self) -> Tuple[List[int], torch.FloatTensor, torch.FloatTensor]:
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = self.collate_fn([self.dataset[i] for i in indices])
        if self.pin_memory:
            batch = pin_memory_batch(batch)
        return [indices]+batch

    def __iter__(self):
        return self


class NatDataLoader(object):

    def __init__(self,
                 dataset: data.Dataset,
                 batch_size: int=1,
                 shuffle: bool=False,
                 sampler: Sampler=None,
                 batch_sampler: Sampler=None,
                 collate_fn: Callable=default_collate,
                 pin_memory: bool=False,
                 drop_last: bool=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return NatDataLoaderIter(self)

    def __len__(self) -> int:
        return len(self.batch_sampler)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    # Dataset already standardized [0;1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = NatCIFAR10(root='./data', train=True,
                          download=True, transform=transform)

    dl_train = NatDataLoader(trainset, batch_size=2, shuffle=True)

    # Test streaming and override targets.
    for i in range(2):
        if i == 0:
            for idx, x, y, nat in dl_train:
                trainset.update_targets(idx, np.zeros(shape=(2, 50)))
        else:
            for idx, x, y, nat in dl_train:
                assert np.array_equal(nat.numpy(), np.zeros(nat.shape, dtype=np.float32))
