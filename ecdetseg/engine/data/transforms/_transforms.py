"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import random
import warnings
from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import InterpolationMode

from ...core import register
from .._misc import (BoundingBoxes, Image, Mask, SanitizeBoundingBoxes, Video,
                     _boxes_keys, convert_to_tv_tensor)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


def _get_hw(inpt):
    size = F.get_spatial_size(inpt)
    return int(size[0]), int(size[1])


def _as_box_tensor(boxes, spatial_size):
    return convert_to_tv_tensor(boxes, key='boxes', box_format='XYXY', spatial_size=spatial_size)


@register()
class ResizeWithKeypoints(T.Transform):
    def __init__(self, size) -> None:
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = tuple(size)

    def forward(self, *inputs):
        image, target = inputs if len(inputs) > 1 else inputs[0]
        old_h, old_w = _get_hw(image)
        image = F.resize(image, self.size)
        new_h, new_w = _get_hw(image)
        target = target.copy()

        ratio_x = float(new_w) / float(old_w)
        ratio_y = float(new_h) / float(old_h)
        if 'boxes' in target:
            boxes = target['boxes'] * torch.as_tensor([ratio_x, ratio_y, ratio_x, ratio_y])
            target['boxes'] = _as_box_tensor(boxes, (new_h, new_w))
        if 'area' in target:
            target['area'] = target['area'] * (ratio_x * ratio_y)
        if 'keypoints' in target:
            keypoints = target['keypoints'].clone()
            keypoints[..., 0] = keypoints[..., 0] * ratio_x
            keypoints[..., 1] = keypoints[..., 1] * ratio_y
            target['keypoints'] = keypoints
        target['size'] = torch.as_tensor([new_h, new_w])
        return image, target


@register()
class RandomHorizontalFlipWithKeypoints(T.Transform):
    _warned_no_pairs = False

    def __init__(self, p=0.5, keypoint_flip_pairs=None, allow_without_flip_pairs=False) -> None:
        super().__init__()
        self.p = float(p)
        self.keypoint_flip_pairs = keypoint_flip_pairs or []
        self.allow_without_flip_pairs = allow_without_flip_pairs

    def forward(self, *inputs):
        image, target = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1).item() >= self.p:
            return image, target
        if 'keypoints' in target and not self.keypoint_flip_pairs and not self.allow_without_flip_pairs:
            if not RandomHorizontalFlipWithKeypoints._warned_no_pairs:
                warnings.warn(
                    'RandomHorizontalFlipWithKeypoints disabled because no keypoint_flip_pairs were supplied.',
                    UserWarning,
                )
                RandomHorizontalFlipWithKeypoints._warned_no_pairs = True
            return image, target

        _, w = _get_hw(image)
        if hasattr(F, 'horizontal_flip'):
            image = F.horizontal_flip(image)
        else:
            image = F.hflip(image)

        target = target.copy()
        if 'boxes' in target:
            boxes = target['boxes']
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
            target['boxes'] = _as_box_tensor(boxes, _get_hw(image))

        if 'keypoints' in target:
            keypoints = target['keypoints'].clone()
            keypoints[..., 0] = float(w) - keypoints[..., 0]
            for left, right in self.keypoint_flip_pairs:
                keypoints[:, [left, right], :] = keypoints[:, [right, left], :]
            target['keypoints'] = keypoints

        if 'masks' in target:
            target['masks'] = target['masks'].flip(-1)
        return image, target


def _image_size_hw(image):
    if isinstance(image, PIL.Image.Image):
        w, h = image.size
        return h, w
    return F.get_spatial_size(image)


def _filter_target(target, keep):
    for key in ["boxes", "labels", "area", "iscrowd", "masks", "keypoints", "keypoint_valid", "has_keypoints"]:
        if key in target:
            target[key] = target[key][keep]
    return target


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertKeypoints(nn.Module):
    def __init__(self, normalize=False, invalidate_outside=True):
        super().__init__()
        self.normalize = normalize
        self.invalidate_outside = invalidate_outside

    def forward(self, sample):
        image, target = sample
        if "keypoints" not in target:
            return image, target
        h, w = _image_size_hw(image)
        keypoints = target["keypoints"].clone()
        if self.invalidate_outside:
            inside = (
                (keypoints[..., 0] >= 0) & (keypoints[..., 0] <= w) &
                (keypoints[..., 1] >= 0) & (keypoints[..., 1] <= h)
            )
            keypoints[..., 2] = torch.where(inside, keypoints[..., 2], torch.zeros_like(keypoints[..., 2]))
        if self.normalize:
            scale = keypoints.new_tensor([w, h])
            keypoints[..., :2] = keypoints[..., :2] / scale
        target["keypoints"] = keypoints
        return image, target


@register()
class KeypointResize(nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True):
        super().__init__()
        self.resize = T.Resize(size=size, interpolation=interpolation, max_size=max_size, antialias=antialias)

    def forward(self, sample):
        image, target = sample
        old_h, old_w = _image_size_hw(image)
        image, target = self.resize(image, target)
        new_h, new_w = _image_size_hw(image)
        if "keypoints" in target:
            keypoints = target["keypoints"].clone()
            keypoints[..., 0] *= float(new_w) / max(float(old_w), 1.0)
            keypoints[..., 1] *= float(new_h) / max(float(old_h), 1.0)
            target["keypoints"] = keypoints
        return image, target


@register()
class KeypointRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5, flip_pairs=None, allow_without_flip_pairs=False):
        super().__init__()
        self.p = p
        self.flip_pairs = flip_pairs or []
        self.allow_without_flip_pairs = allow_without_flip_pairs
        if not self.flip_pairs and self.p > 0 and not allow_without_flip_pairs:
            warnings.warn(
                "KeypointRandomHorizontalFlip has no flip_pairs; horizontal flipping will be disabled.",
                UserWarning,
            )

    def forward(self, sample):
        image, target = sample
        if random.random() >= self.p:
            return image, target
        if "keypoints" in target and not self.flip_pairs and not self.allow_without_flip_pairs:
            return image, target

        h, w = _image_size_hw(image)
        image = F.horizontal_flip(image)
        if "boxes" in target:
            boxes = target["boxes"]
            flipped = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1], dtype=boxes.dtype, device=boxes.device) \
                + torch.as_tensor([w, 0, w, 0], dtype=boxes.dtype, device=boxes.device)
            target["boxes"] = convert_to_tv_tensor(flipped, key="boxes", box_format="xyxy", spatial_size=(h, w))
        if "masks" in target:
            target["masks"] = target["masks"].flip(-1)
        if "keypoints" in target:
            keypoints = target["keypoints"].clone()
            valid = keypoints[..., 2] > 0
            keypoints[..., 0] = torch.where(valid, keypoints.new_tensor(float(w)) - keypoints[..., 0], keypoints[..., 0])
            for left, right in self.flip_pairs:
                keypoints[:, [left, right], :] = keypoints[:, [right, left], :]
            target["keypoints"] = keypoints
        return image, target


@register()
class KeypointSanitizeBoundingBoxes(nn.Module):
    def __init__(self, min_size=1):
        super().__init__()
        self.min_size = min_size

    def forward(self, sample):
        image, target = sample
        if "boxes" not in target:
            return image, target
        boxes = target["boxes"]
        keep = (boxes[:, 2] - boxes[:, 0] >= self.min_size) & (boxes[:, 3] - boxes[:, 1] >= self.min_size)
        return image, _filter_target(target, keep)


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
