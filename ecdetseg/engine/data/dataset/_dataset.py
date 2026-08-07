"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data as data

from ...core import register


class DetDataset(data.Dataset):
    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        raise NotImplementedError("Please implement this function to return item before `transforms`.")

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1


@register()
class WeightedMultiDataset(data.Dataset):
    """Sample several map-style datasets according to dataset-level weights.

    The virtual index is mapped deterministically to a dataset and local sample.
    Consequently PyTorch's normal ``DistributedSampler`` can shard this dataset
    without changing the requested mixture across ranks.
    """

    __inject__ = ['datasets', 'transforms']

    def __init__(self, datasets, weights, transforms=None, samples_per_epoch=None, seed=0):
        if not datasets:
            raise ValueError('WeightedMultiDataset requires at least one dataset')
        if len(datasets) != len(weights):
            raise ValueError(f'datasets ({len(datasets)}) and weights ({len(weights)}) must have equal length')
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError('WeightedMultiDataset does not support empty component datasets')

        weights = torch.as_tensor(weights, dtype=torch.double)
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('weights must be finite, non-negative, and contain at least one positive value')

        self.datasets = list(datasets)
        self.transforms = transforms
        self.weights = weights / weights.sum()
        self.samples_per_epoch = int(samples_per_epoch or sum(len(dataset) for dataset in datasets))
        if self.samples_per_epoch <= 0:
            raise ValueError('samples_per_epoch must be positive')
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch * self.samples_per_epoch + index)
        dataset_index = int(torch.multinomial(self.weights, 1, generator=generator).item())
        local_index = int(torch.randint(len(self.datasets[dataset_index]), (1,), generator=generator).item())
        image, target = self.datasets[dataset_index][local_index]
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def set_epoch(self, epoch):
        self._epoch = int(epoch)
        if self.transforms is not None and hasattr(self.transforms, 'set_epoch'):
            self.transforms.set_epoch(epoch)
        for dataset in self.datasets:
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch
