"""
Detection-equivalence integration test scaffold.

This test requires local ECDet S/M/L/X checkpoints. It is skipped unless
ECDETPOSE_EQUIV_CHECKPOINT_DIR points at a directory containing:
ecdet_s.pth, ecdet_m.pth, ecdet_l.pth, ecdet_x.pth.
"""

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get("ECDETPOSE_EQUIV_CHECKPOINT_DIR"), reason="requires ECDet checkpoints")
def test_ecdetpose_detection_equivalence():
    import sys
    import torch

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from engine.core import YAMLConfig

    ckpt_dir = Path(os.environ["ECDETPOSE_EQUIV_CHECKPOINT_DIR"])
    rows = []
    for variant in ["s", "m", "l", "x"]:
        ecdet = YAMLConfig(str(root / "configs" / "ecdet" / f"ecdet_{variant}.yml")).model.eval()
        pose = YAMLConfig(str(root / "configs" / "ecdetpose" / f"ecdetpose_{variant}.yml")).model.eval()
        state = torch.load(ckpt_dir / f"ecdet_{variant}.pth", map_location="cpu")
        state = state.get("model", state)
        ecdet.load_state_dict(state, strict=True)
        pose_state = pose.state_dict()
        pose_state.update({k: v for k, v in state.items() if k in pose_state and pose_state[k].shape == v.shape})
        pose.load_state_dict(pose_state, strict=True)
        x = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            out_det = ecdet(x)
            out_pose = pose(x)
        logit_diff = (out_det["pred_logits"] - out_pose["pred_logits"]).abs().max().item()
        box_diff = (out_det["pred_boxes"] - out_pose["pred_boxes"]).abs().max().item()
        rows.append((variant, logit_diff, box_diff))
        assert logit_diff <= 1e-5
        assert box_diff <= 1e-5
    print("variant max_logit_difference max_bbox_difference passed")
    for row in rows:
        print(row[0], row[1], row[2], True)
