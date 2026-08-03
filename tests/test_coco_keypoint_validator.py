import json

import pytest

pytest.importorskip("PIL")

from ecdetseg.tools.data.validate_coco_keypoints import validate
from tests.utils.create_tiny_vehicle_keypoint_coco import create_tiny_vehicle_keypoint_coco


def test_validator_reports_tiny_dataset(tmp_path):
    _, ann_file = create_tiny_vehicle_keypoint_coco(tmp_path)
    dataset = json.loads(ann_file.read_text(encoding="utf-8"))
    report = validate(dataset, num_keypoints=31)
    assert report["number_of_images"] == 4
    assert report["number_of_keypoints"] == 31
    assert report["empty_images"] == 1
    assert report["malformed_keypoint_arrays"] == []


def test_validator_detects_malformed_keypoints(tmp_path):
    _, ann_file = create_tiny_vehicle_keypoint_coco(tmp_path)
    dataset = json.loads(ann_file.read_text(encoding="utf-8"))
    dataset["annotations"][0]["keypoints"] = [1, 2, 2]
    report = validate(dataset, num_keypoints=31)
    assert report["malformed_keypoint_arrays"]
