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
    __share__ = ['num_classes']

    def __init__(self, datasets, weights, transforms=None, samples_per_epoch=None,
                 seed=0, category_names=None, remap_categories_by_name=True,
                 num_classes=None):
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
        self.category_names = self._configure_category_mapping(
            category_names, remap_categories_by_name)
        if num_classes is not None and self.category_names is not None \
                and len(self.category_names) != int(num_classes):
            raise ValueError(
                f'Weighted dataset defines {len(self.category_names)} categories '
                f'{self.category_names}, but num_classes={num_classes}')
        self.transforms = transforms
        # CocoDetection and the solver historically expose the composed
        # augmentation pipeline as ``_transforms``. Keep that dataset
        # interface on the weighted wrapper as well as the public alias.
        self._transforms = transforms
        self.weights = weights / weights.sum()
        self.samples_per_epoch = int(samples_per_epoch or sum(len(dataset) for dataset in datasets))
        if self.samples_per_epoch <= 0:
            raise ValueError('samples_per_epoch must be positive')
        self.seed = int(seed)
        self._epoch = 0

    def _configure_category_mapping(self, category_names, enabled):
        if not enabled:
            return list(category_names) if category_names is not None else None
        if category_names is None:
            first_categories = getattr(self.datasets[0], 'categories', None)
            if first_categories is None:
                raise ValueError(
                    'category_names is required when a component dataset has no COCO categories')
            category_names = [category['name'] for category in first_categories]
        category_names = list(category_names)
        if len(category_names) != len(set(category_names)):
            raise ValueError(f'category_names contains duplicates: {category_names}')
        for dataset in self.datasets:
            setter = getattr(dataset, 'set_category_name_mapping', None)
            if setter is None:
                raise TypeError(
                    f'{type(dataset).__name__} cannot remap categories by name')
            setter(category_names)
        return category_names

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
        target['dataset_index'] = torch.tensor(dataset_index, dtype=torch.int64)
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
