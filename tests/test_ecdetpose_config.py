import os
import sys

import pytest


torch = pytest.importorskip("torch")


def test_ecdetpose_s_builds_and_outputs_keypoints():
    repo = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(repo, "ecdetseg"))
    import engine  # noqa: F401
    from engine.core import YAMLConfig

    cfg = YAMLConfig(
        os.path.join(repo, "ecdetseg", "configs", "ecdetpose", "ecdetpose_s.yml"),
        ViTAdapter={"skip_load_backbone": True, "weights_path": None},
        eval_spatial_size=[256, 256],
        ECTransformer={"num_queries": 16, "num_denoising": 0},
    )
    model = cfg.model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    assert out["pred_logits"].shape[:2] == (1, 16)
    assert out["pred_boxes"].shape == (1, 16, 4)
    assert out["pred_keypoints"].shape == (1, 16, 31, 2)
    assert out["pred_keypoint_logits"].shape == (1, 16, 31)
    boxes = out["pred_boxes"]
    kpts = out["pred_keypoints"]
    xy_min = boxes[..., :2] - 0.5 * boxes[..., 2:]
    xy_max = boxes[..., :2] + 0.5 * boxes[..., 2:]
    assert torch.all(kpts >= xy_min[:, :, None, :] - 1e-6)
    assert torch.all(kpts <= xy_max[:, :, None, :] + 1e-6)
