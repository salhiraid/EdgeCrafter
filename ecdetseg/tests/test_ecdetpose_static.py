import json
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PIL")

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
    report = json.loads(result.stdout)
    assert report["number_of_keypoints"] == 31
    assert report["malformed_keypoint_arrays"] == []
    assert report["empty_images"] == 1


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
    report = json.loads(result.stdout)
    assert len(report["malformed_keypoint_arrays"]) == 1


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


def test_five_dataset_weighted_config():
    yaml = pytest.importorskip("yaml")
    config_path = ROOT / "configs" / "ecdetpose" / "examples" / "ecdetpose_s_vehicle_5datasets.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["train_dataloader"]["dataset"]
    assert dataset["type"] == "WeightedMultiDataset"
    assert len(dataset["datasets"]) == 5
    assert len(dataset["weights"]) == 5
    assert sum(dataset["weights"]) == pytest.approx(1.0)
    assert dataset["samples_per_epoch"] > 0
    assert dataset["remap_categories_by_name"] is True


def test_five_dataset_include_replaces_coco_dataset_node():
    pytest.importorskip("yaml")
    core_root = ROOT / "engine" / "core"
    package = types.ModuleType("_config_test_engine")
    package.__path__ = []
    sys.modules[package.__name__] = package
    core_package = types.ModuleType("_config_test_engine.core")
    core_package.__path__ = [str(core_root)]
    sys.modules[core_package.__name__] = core_package
    for module_name in ("workspace", "yaml_utils"):
        full_name = f"_config_test_engine.core.{module_name}"
        spec = importlib.util.spec_from_file_location(full_name, core_root / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    load_config = sys.modules["_config_test_engine.core.yaml_utils"].load_config
    config = load_config(str(ROOT / "configs" / "ecdetpose" / "examples" /
                             "ecdetpose_s_vehicle_5datasets.yml"))
    dataset = config["train_dataloader"]["dataset"]
    assert dataset["type"] == "WeightedMultiDataset"
    assert not {"img_folder", "ann_file", "num_keypoints"}.intersection(dataset)


def test_weighted_dataset_matches_solver_transform_interface():
    dataset_source = (ROOT / "engine" / "data" / "dataset" / "_dataset.py").read_text(
        encoding="utf-8")
    solver_source = (ROOT / "engine" / "solver" / "ec_solver.py").read_text(
        encoding="utf-8")
    assert "self._transforms = transforms" in dataset_source
    assert "getattr(train_dataset, '_transforms', None)" in solver_source
    assert "getattr(train_dataset, 'transforms', None)" in solver_source


def test_weighted_dataset_remaps_and_validates_category_labels():
    dataset_source = (ROOT / "engine" / "data" / "dataset" / "_dataset.py").read_text(
        encoding="utf-8")
    coco_source = (ROOT / "engine" / "data" / "dataset" / "coco_dataset.py").read_text(
        encoding="utf-8")
    engine_source = (ROOT / "engine" / "solver" / "ec_engine.py").read_text(
        encoding="utf-8")
    denoising_source = (ROOT / "engine" / "edgecrafter" / "denoising.py").read_text(
        encoding="utf-8")
    assert "remap_categories_by_name=True" in dataset_source
    assert "__share__ = ['num_classes']" in dataset_source
    assert "set_category_name_mapping" in coco_source
    assert "_validate_target_labels(targets, criterion.num_classes)" in engine_source
    assert "Denoising target labels must be in" in denoising_source


def test_vehicle_config_enables_coco_bbox_and_keypoint_metrics():
    yaml = pytest.importorskip("yaml")
    config_path = ROOT / "configs" / "ecdetpose" / "examples" / "ecdetpose_s_vehicle_31kpts.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluator = config["evaluator"]
    assert evaluator["iou_types"] == ["bbox", "keypoints"]
    assert len(evaluator["keypoint_oks_sigmas"]) == 31
    assert evaluator["keypoint_score_mode"] == "bbox_keypoint"
    assert evaluator["keypoint_score_thr"] == pytest.approx(0.2)
    assert evaluator["keypoint_distance_thresholds"] == [5, 10]
    assert evaluator["keypoint_visibility_thr"] == pytest.approx(0.5)
    assert evaluator["pose_detection_score_thr"] == pytest.approx(0.3)
    assert evaluator["pose_crop_size"] == 512
    assert evaluator["pose_crop_margin"] == pytest.approx(0.05)
    assert evaluator["pose_min_bbox_size"] == 128


def test_tensorboard_uses_named_coco_metrics():
    solver_source = (ROOT / "engine" / "solver" / "ec_solver.py").read_text(encoding="utf-8")
    evaluator_source = (ROOT / "engine" / "data" / "dataset" / "coco_eval.py").read_text(encoding="utf-8")
    assert "Performance/BBox" in solver_source
    assert "Performance/Keypoints_COCO_OKS" in solver_source
    assert "Performance/Keypoints_Pixel" in solver_source
    assert "{tensorboard_group}/{metric_name}" in solver_source
    assert "Performance/Keypoints_PerJoint" in solver_source
    for name in ("AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_medium", "AR_large"):
        assert repr(name) in evaluator_source


def test_pose_metrics_and_prediction_visualizations_are_wired():
    engine_source = (ROOT / "engine" / "solver" / "ec_engine.py").read_text(encoding="utf-8")
    evaluator_source = (ROOT / "engine" / "data" / "dataset" / "coco_eval.py").read_text(encoding="utf-8")
    solver_source = (ROOT / "engine" / "solver" / "ec_solver.py").read_text(encoding="utf-8")
    for name in ("Precision_", "Recall_", "F1_", "Visibility_Precision", "Visibility_Recall"):
        assert name in evaluator_source
    assert "prediction_visualizations" in engine_source
    assert "_load_original_validation_image" in engine_source
    assert "validation_image_root" in engine_source
    assert "max_visualizations=10" in solver_source
    assert "color = (255, 64, 64) if confident else (255, 165, 0)" in engine_source
    assert "self._write_eval_metrics(test_stats" in solver_source


def test_default_pose_metric_names_are_registered():
    evaluator_source = (ROOT / "engine" / "data" / "dataset" / "coco_eval.py").read_text(encoding="utf-8")
    for name in (
            "Precision_5px", "Recall_5px", "F1_5px",
            "Precision_10px", "Recall_10px", "F1_10px"):
        assert repr(name) in evaluator_source
    assert "Visibility_Accuracy" in evaluator_source
    assert "pose_per_keypoint_metrics" in evaluator_source


def test_pose_configs_use_keypoint_aware_box_sanitizer():
    config_paths = (
        ROOT / "configs" / "ecdetpose" / "ecdetpose.yml",
        ROOT / "configs" / "ecdetpose" / "examples" / "ecdetpose_s_vehicle_31kpts.yml",
        ROOT / "configs" / "ecdetpose" / "examples" / "ecdetpose_s_vehicle_5datasets.yml",
    )
    for config_path in config_paths:
        source = config_path.read_text(encoding="utf-8")
        assert "type: KeypointSanitizeBoundingBoxes" in source
        assert "type: SanitizeBoundingBoxes" not in source

    matcher_source = (ROOT / "engine" / "edgecrafter" / "matcher.py").read_text(
        encoding="utf-8")
    assert "Target field alignment error" in matcher_source


def test_pose_accuracy_recipe_configs():
    yaml = pytest.importorskip("yaml")
    examples = ROOT / "configs" / "ecdetpose" / "examples"
    for size in ("s", "m"):
        for recipe in ("balanced", "precision", "finetune"):
            path = examples / f"ecdetpose_{size}_vehicle_pose_{recipe}.yml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            criterion = config["ECCriterion"]
            assert criterion["weight_dict"]["loss_keypoint"] >= 10
            assert criterion["matcher"]["weight_dict"]["keypoint_cost_weight"] > 0
            assert criterion["matcher"]["weight_dict"]["oks_cost_weight"] == 0
        precision = yaml.safe_load(
            (examples / f"ecdetpose_{size}_vehicle_pose_precision.yml").read_text())
        assert precision["eval_spatial_size"] == [960, 960]
        assert precision["train_dataloader"]["collate_fn"]["mixup_prob"] == 0.0


def test_pose_loss_and_matcher_ablation_configs():
    yaml = pytest.importorskip("yaml")
    examples = ROOT / "configs" / "ecdetpose" / "examples"
    expected = {
        "coordinate": (20, 2, 1, 6, 0),
        "oks": (8, 10, 1, 1, 4),
        "visibility": (10, 4, 3, 2, 0),
    }
    for size in ("s", "m"):
        for recipe, values in expected.items():
            path = examples / f"ecdetpose_{size}_vehicle_pose_{recipe}.yml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            weights = config["ECCriterion"]["weight_dict"]
            matcher = config["ECCriterion"]["matcher"]
            costs = matcher["weight_dict"]
            assert (weights["loss_keypoint"], weights["loss_oks"],
                    weights["loss_keypoint_visibility"],
                    costs["keypoint_cost_weight"],
                    costs["oks_cost_weight"]) == values
            if recipe == "oks":
                assert len(matcher["keypoint_oks_sigmas"]) == 31


def test_matcher_oks_uses_normalized_box_area_and_configured_sigmas():
    source = (ROOT / "engine" / "edgecrafter" / "matcher.py").read_text(
        encoding="utf-8")
    assert "areas = (tgt_bbox[:, 2] * tgt_bbox[:, 3])" in source
    assert "self.keypoint_oks_sigmas" in source
    assert "cost_oks[:, ~pose_valid] = 0.0" in source
