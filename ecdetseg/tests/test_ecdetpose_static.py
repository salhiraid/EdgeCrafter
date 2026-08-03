import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.create_tiny_vehicle_keypoint_coco import create_tiny_vehicle_keypoint_coco


ROOT = Path(__file__).resolve().parents[1]


def test_tiny_vehicle_keypoint_coco_validator(tmp_path):
    image_dir, ann_file = create_tiny_vehicle_keypoint_coco(tmp_path, num_images=4, num_keypoints=31)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "data" / "validate_coco_keypoints.py"),
            "--ann-file",
            str(ann_file),
            "--image-root",
            str(image_dir),
            "--num-keypoints",
            "31",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "num_keypoints: 31" in result.stdout
    assert "malformed_keypoint_arrays: 0" in result.stdout
    assert "empty_images: 1" in result.stdout


def test_malformed_keypoint_annotation_is_reported(tmp_path):
    _, ann_file = create_tiny_vehicle_keypoint_coco(tmp_path, num_images=2, num_keypoints=31)
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    data["annotations"][0]["keypoints"] = data["annotations"][0]["keypoints"][:-1]
    ann_file.write_text(json.dumps(data), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "data" / "validate_coco_keypoints.py"),
            "--ann-file",
            str(ann_file),
            "--num-keypoints",
            "31",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "malformed_keypoint_arrays: 1" in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires PyTorch runtime")
def test_ecdetpose_model_construction_requires_torch():
    import os

    sys.path.insert(0, str(ROOT))
    from engine.core import YAMLConfig

    for variant in ["s", "m", "l", "x"]:
        cfg = YAMLConfig(str(ROOT / "configs" / "ecdetpose" / f"ecdetpose_{variant}.yml"))
        model = cfg.model
        assert type(model).__name__ == "ECDetPose"
        assert model.decoder.num_keypoints == 31
