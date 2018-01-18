#!/usr/bin/python
import numpy as np
from typing import List, Tuple

from torchvision.datasets import ImageFolder

from core.dataloader import DataLoader

from datasets.utils import generate_random_targets
from datasets.interface import DynamicTargetDataset


class NatImageFolder(ImageFolder, DynamicTargetDataset):
    """ Override ImageFolder Dataset to also stream the generated encoder target alongside it's input and label.

        Args:
            z_dims (int): The dimensionality of the latent space
            root (string): Root directory path.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            loader (callable, optional): A function to load an image given its path.
         Attributes:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            imgs (list): List of (image path, class_index) tuples
        """

    def __init__(self, z_dims: int=50, **kwargs) -> None:
        super().__init__(**kwargs)
        self.z_dims = z_dims

        self.nat_targets = generate_random_targets(len(self.imgs), z_dims)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = super().__getitem__(index)
        return x, y, self.nat_targets[index, :]

    def update_targets(self, indexes: List[int], new_targets: np.ndarray):
        """
        Helper method that update the assigned feature representation
        Used after cost minimisation every few epoch.
        :param indexes:
        :param new_targets:
        :return:
        """
        self.nat_targets[indexes, :] = new_targets


if __name__ == '__main__':
    import torchvision.transforms as transforms

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('L')),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Does [0; 1] Standardisation
    ])

    z_dims = 50
    batch_size = 2
    traindir = './data/imagenet_test'

    trainset = NatImageFolder(root=traindir, z_dims=z_dims, transform=transform)

    dl_train = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=False)

    # Test streaming and override targets.
    for i in range(2):
        if i == 0:
            for idx, x, y, nat in dl_train:
                trainset.update_targets(idx, np.zeros(shape=(batch_size, z_dims)))
        else:
            for idx, x, y, nat in dl_train:
                assert np.array_equal(nat.numpy(), np.zeros(nat.shape, dtype=np.float32))
