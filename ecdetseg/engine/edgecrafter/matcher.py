"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..core import register
from .box_ops import (batch_dice_loss, batch_sigmoid_ce_loss,
                      box_cxcywh_to_xyxy, generalized_box_iou)
from .segmentation_head import point_sample


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                 mask_point_sample_ratio=None, keypoint_oks_sigmas=None,
                 **kwargs):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.cost_keypoint = weight_dict.get("keypoint_cost_weight", weight_dict.get("cost_keypoint", 0.0))
        self.cost_oks = weight_dict.get("oks_cost_weight", weight_dict.get("cost_oks", 0.0))
        self.keypoint_oks_sigmas = keypoint_oks_sigmas

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        
        self.mask_point_sample_ratio = mask_point_sample_ratio
        if self.mask_point_sample_ratio:
            self.cost_mask_ce = weight_dict["cost_mask_ce"]
            self.cost_mask_dice = weight_dict["cost_mask_dice"]
        
        

        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Every per-instance field must remain aligned with boxes. In pose
        # pipelines torchvision's generic SanitizeBoundingBoxes can remove an
        # invalid box without filtering custom keypoint tensors, which used to
        # fail later with an opaque cost-matrix broadcasting error.
        for batch_index, target in enumerate(targets):
            num_targets = len(target["boxes"])
            for field in ("labels", "keypoints", "keypoint_valid", "has_keypoints"):
                if field in target and len(target[field]) != num_targets:
                    image_id = target.get("image_id", "<unknown>")
                    if torch.is_tensor(image_id):
                        image_id = image_id.flatten().tolist()
                    raise ValueError(
                        f"Target field alignment error at batch_index={batch_index}, "
                        f"image_id={image_id}: boxes has {num_targets} instances but "
                        f"{field} has {len(target[field])}. Every transform and "
                        "collate augmentation must apply the same keep/concat operation "
                        "to all instance fields; use the pose-aware sanitizer and MixUp.")

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        cost_keypoint = None
        cost_oks = None
        if (self.cost_keypoint or self.cost_oks) and 'pred_keypoints' in outputs and all('keypoints' in v for v in targets):
            out_keypoints = outputs['pred_keypoints'].flatten(0, 1)
            tgt_keypoints = torch.cat([v['keypoints'] for v in targets]).to(out_keypoints.device)
            if tgt_keypoints.numel() > 0:
                instance_valid = torch.cat([
                    v.get('keypoint_valid', v.get('has_keypoints', torch.ones(
                        len(v['boxes']), dtype=torch.bool, device=v['boxes'].device)))
                    for v in targets
                ]).to(device=out_keypoints.device, dtype=torch.bool)
                valid = (tgt_keypoints[..., 2] > 0) & instance_valid[:, None]
                distance = (out_keypoints[:, None] - tgt_keypoints[None, :, :, :2]).abs().sum(-1)
                valid_f = valid[None].to(distance.dtype)
                cost_keypoint = (distance * valid_f).sum(-1) / valid_f.sum(-1).clamp(min=1)

                # Keypoints and boxes have both been normalized by the pose
                # pipeline, so use normalized bbox area here. COCO's raw pixel
                # area would make the OKS matching cost almost constant.
                areas = (tgt_bbox[:, 2] * tgt_bbox[:, 3]).to(out_keypoints.device).clamp(min=1e-6)
                squared_distance = ((out_keypoints[:, None] - tgt_keypoints[None, :, :, :2]) ** 2).sum(-1)
                if self.keypoint_oks_sigmas is None:
                    sigmas = out_keypoints.new_full((tgt_keypoints.shape[1],), 0.1)
                else:
                    if len(self.keypoint_oks_sigmas) != tgt_keypoints.shape[1]:
                        raise ValueError(
                            'HungarianMatcher keypoint_oks_sigmas must contain '
                            f'{tgt_keypoints.shape[1]} values, got '
                            f'{len(self.keypoint_oks_sigmas)}')
                    sigmas = out_keypoints.new_tensor(self.keypoint_oks_sigmas)
                variances = (sigmas * 2) ** 2
                denom = areas[None, :, None] * variances[None, None, :] * 2.0
                oks = torch.exp(-squared_distance / denom.clamp(min=1e-12))
                oks = (oks * valid_f).sum(-1) / valid_f.sum(-1).clamp(min=1)
                cost_oks = 1.0 - oks
                # Bbox-only instances have no meaningful pose matching cost.
                pose_valid = valid.any(dim=1)
                cost_keypoint[:, ~pose_valid] = 0.0
                cost_oks[:, ~pose_valid] = 0.0
        
        masks_present = "masks" in targets[0] and 'pred_masks' in outputs
        if masks_present:
            tgt_masks = torch.cat([v["masks"] for v in targets])
            out_masks = outputs["pred_masks"].flatten(0, 1)
            # Resize predicted masks to target mask size if needed
            # if out_masks.shape[-2:] != tgt_masks.shape[-2:]:
            #     # out_masks = F.interpolate(out_masks.unsqueeze(1), size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            #     tgt_masks = F.interpolate(tgt_masks.unsqueeze(1).float(), size=out_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            # # Flatten masks
            # pred_masks_logits = out_masks.flatten(1)  # [P, HW]
            # tgt_masks_flat = tgt_masks.flatten(1).float()  # [T, HW]

            num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio

            tgt_masks = tgt_masks.to(out_masks.dtype)

            point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
            pred_masks_logits = point_sample(out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1), align_corners=False).squeeze(1)
            tgt_masks_flat = point_sample(tgt_masks.unsqueeze(1), point_coords.repeat(tgt_masks.shape[0], 1, 1), align_corners=False, mode="nearest").squeeze(1)

            # Binary cross-entropy with logits cost (mean over pixels), computed pairwise efficiently
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)

            # Dice loss cost (1 - dice coefficient)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)
            
        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if cost_keypoint is not None:
            C = C + self.cost_keypoint * cost_keypoint
        if cost_oks is not None:
            C = C + self.cost_oks * cost_oks
        if masks_present:
            C = C + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        # C.copy_(C_original)
        return indices_list
