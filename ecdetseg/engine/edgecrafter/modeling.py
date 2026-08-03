"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import torch.nn as nn

from ..core import register

__all__ = ['ECDet', 'ECSeg', 'ECDetPose']


class _ECBase(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.encoder(x)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@register()
class ECDet(_ECBase):

    def forward(self, x, targets=None):
        x = self.forward_features(x)
        return self.decoder(x, targets)


@register()
class ECDetPose(_ECBase):
    __share__ = ['num_keypoints', 'constrain_keypoints_to_box']

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_keypoints=31,
        constrain_keypoints_to_box=True,
        keypoint_head_layers=3,
        keypoint_visibility_bias=0.0,
    ):
        super().__init__(backbone, encoder, decoder)
        self.num_keypoints = int(num_keypoints)
        self.constrain_keypoints_to_box = bool(constrain_keypoints_to_box)
        if hasattr(self.decoder, 'num_keypoints'):
            self.decoder.num_keypoints = self.num_keypoints
        if hasattr(self.decoder, 'constrain_keypoints_to_box'):
            self.decoder.constrain_keypoints_to_box = self.constrain_keypoints_to_box

    def forward(self, x, targets=None):
        x = self.forward_features(x)
        out = self.decoder(x, targets)
        if 'pred_keypoints' not in out or 'pred_keypoint_logits' not in out:
            raise RuntimeError('ECDetPose decoder must be configured with ECTransformer.num_keypoints > 0.')
        return out


@register()
class ECSeg(_ECBase):

    def forward(self, x, targets=None):
        x = self.forward_features(x)
        spatial_feat = x[0]
        return self.decoder(x, targets, spatial_feat)
