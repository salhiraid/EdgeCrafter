"""Keypoint-aware transforms used by the optional ECDet pose decoder."""

import random

import torch
import torchvision.transforms.functional as F
from PIL import Image

from ...core import register


def _unpack(inputs):
    return inputs[0] if len(inputs) == 1 and isinstance(inputs[0], tuple) else inputs


@register()
class PoseResize:
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, *inputs):
        image, target = _unpack(inputs)
        old_w, old_h = image.size
        new_h, new_w = self.size
        image = F.resize(image, [new_h, new_w])
        target = target.copy()
        scale = torch.tensor([new_w / old_w, new_h / old_h, 1.0])
        target["boxes"] = target["boxes"] * torch.tensor(
            [new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h])
        if "keypoints" in target:
            target["keypoints"] = target["keypoints"] * scale
        target["size"] = torch.tensor([new_h, new_w])
        return image, target


@register()
class PoseHorizontalFlip:
    def __init__(self, p=0.5, flip_pairs=None):
        self.p = p
        self.flip_pairs = flip_pairs or []

    def __call__(self, *inputs):
        image, target = _unpack(inputs)
        if random.random() >= self.p:
            return image, target
        width, _ = image.size
        image, target = F.hflip(image), target.copy()
        boxes = target["boxes"]
        target["boxes"] = boxes[:, [2, 1, 0, 3]] * boxes.new_tensor([-1, 1, -1, 1]) + boxes.new_tensor([width, 0, width, 0])
        if "keypoints" in target:
            points = target["keypoints"].clone()
            points[..., 0] = torch.where(points[..., 2] > 0, width - points[..., 0] - 1, 0)
            for left, right in self.flip_pairs:
                points[:, [left, right]] = points[:, [right, left]]
            target["keypoints"] = points
        return image, target


@register()
class PoseToTensor:
    def __call__(self, *inputs):
        image, target = _unpack(inputs)
        return F.to_tensor(image), target


@register()
class PoseNormalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, *inputs):
        image, target = _unpack(inputs)
        image = F.normalize(image, self.mean, self.std)
        height, width = image.shape[-2:]
        target = target.copy()
        boxes = target["boxes"]
        boxes = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2,
                             (boxes[:, 1] + boxes[:, 3]) / 2,
                             boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), dim=1)
        target["boxes"] = boxes / boxes.new_tensor([width, height, width, height])
        target["area"] = target["area"] / float(width * height)
        if "keypoints" in target:
            points = target["keypoints"]
            visibility = (points[..., 2] > 0).to(points.dtype)
            xy = points[..., :2] / points.new_tensor([width, height])
            xy = torch.where(visibility[..., None] > 0, xy, 0)
            target["keypoints"] = torch.cat((xy.flatten(1), visibility), dim=1)
        return image, target
