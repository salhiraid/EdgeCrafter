import os
import sys

import pytest


torch = pytest.importorskip("torch")


def test_keypoint_losses_are_finite_with_empty_targets():
    repo = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(repo, "ecdetseg"))
    import engine  # noqa: F401
    from engine.edgecrafter.criterion import ECCriterion
    from engine.edgecrafter.matcher import HungarianMatcher

    matcher = HungarianMatcher({"cost_class": 2, "cost_bbox": 5, "cost_giou": 2})
    criterion = ECCriterion(
        matcher=matcher,
        weight_dict={"loss_keypoint": 1, "loss_oks": 1, "loss_keypoint_visibility": 1},
        losses=["keypoints"],
        num_classes=1,
        num_keypoints=31,
        keypoint_oks_sigmas=[0.1] * 31,
    )
    outputs = {
        "pred_logits": torch.randn(1, 4, 1),
        "pred_boxes": torch.rand(1, 4, 4),
        "pred_keypoints": torch.rand(1, 4, 31, 2),
        "pred_keypoint_logits": torch.randn(1, 4, 31),
        "aux_outputs": [],
        "enc_aux_outputs": [],
    }
    targets = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4), "keypoints": torch.zeros(0, 31, 3)}]
    losses = criterion(outputs, targets)
    assert all(torch.isfinite(v) for v in losses.values())
