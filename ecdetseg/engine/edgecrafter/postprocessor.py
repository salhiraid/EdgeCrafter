"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register

__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        mask_pred = outputs.get('pred_masks', None)
        keypoint_pred = outputs.get('pred_keypoints', None)
        keypoint_logits = outputs.get('pred_keypoint_logits', None)

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            masks = mask_pred.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mask_pred.shape[-2], 
                                                                                           mask_pred.shape[-1])) if mask_pred is not None else None
            keypoints = keypoint_pred.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, keypoint_pred.shape[-2], keypoint_pred.shape[-1])) if keypoint_pred is not None else None
            keypoint_scores = keypoint_logits.sigmoid().gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, keypoint_logits.shape[-1])) if keypoint_logits is not None else None

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
                keypoints = torch.gather(keypoint_pred, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).tile(1, 1, keypoint_pred.shape[-2], keypoint_pred.shape[-1])) if keypoint_pred is not None else None
                keypoint_scores = torch.gather(keypoint_logits.sigmoid(), dim=1, index=index.unsqueeze(-1).tile(1, 1, keypoint_logits.shape[-1])) if keypoint_logits is not None else None
            else:
                keypoints = keypoint_pred
                keypoint_scores = keypoint_logits.sigmoid() if keypoint_logits is not None else None

        if keypoints is not None:
            keypoints = keypoints * orig_target_sizes.unsqueeze(1).unsqueeze(1)

        if self.deploy_mode:
            if mask_pred is not None:
                return labels, boxes, scores, masks
            if keypoints is not None:
                return labels, boxes, scores, keypoints, keypoint_scores
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        if mask_pred is not None:
            for (i, (s, l, b, m)) in enumerate(zip(scores, labels, boxes, masks)):
                res = {'scores': s, 'labels': l, 'boxes': b}
                w, h = orig_target_sizes[i].tolist()
                m = F.interpolate(m.unsqueeze(1), size=(int(h), int(w)), mode='bilinear', align_corners=False)
                res['masks'] = m > 0.0
                results.append(res)
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        if keypoints is not None:
            for i, res in enumerate(results):
                res['keypoints'] = keypoints[i]
                res['keypoint_scores'] = keypoint_scores[i]

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
