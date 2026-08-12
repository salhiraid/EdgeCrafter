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
    assert "tgt_bbox[pose_valid, 2] * tgt_bbox[pose_valid, 3]" in source
    assert "self.keypoint_oks_sigmas" in source
    assert "cost_oks = out_keypoints.new_zeros" in source
    assert "cost_oks[:, pose_valid] = 1.0 - oks" in source


def test_matcher_does_not_evaluate_bbox_only_pose_columns():
    matcher_source = (ROOT / "engine" / "edgecrafter" / "matcher.py").read_text(
        encoding="utf-8")
    dataset_source = (
        ROOT / "engine" / "data" / "dataset" / "coco_dataset.py"
    ).read_text(encoding="utf-8")
    assert "pose_valid = valid.any(dim=1)" in matcher_source
    assert "pose_targets = tgt_keypoints[pose_valid]" in matcher_source
    assert "pose_joint_valid = valid[pose_valid]" in matcher_source
    assert "cost_keypoint[:, pose_valid] = pose_cost_keypoint" in matcher_source
    assert "bool(obj.get(\"keypoints\"))" in dataset_source


def test_matcher_requires_pose_predictions_when_pose_costs_are_enabled():
    matcher_source = (ROOT / "engine" / "edgecrafter" / "matcher.py").read_text(
        encoding="utf-8")
    criterion_source = (ROOT / "engine" / "edgecrafter" / "criterion.py").read_text(
        encoding="utf-8")
    assert "if 'pred_keypoints' not in outputs:" in matcher_source
    assert "outputs does not contain 'pred_keypoints'" in matcher_source
    assert "'pose_costs_used'" in matcher_source
    assert "use_keypoint_costs=False" in criterion_source
    assert "for i, aux_outputs in enumerate(outputs['aux_outputs']):" in criterion_source
    assert "outputs['pre_outputs'], targets," in criterion_source


def test_legacy_sanitizer_and_mixup_keep_pose_fields_aligned():
    transforms_source = (
        ROOT / "engine" / "data" / "transforms" / "_transforms.py"
    ).read_text(encoding="utf-8")
    dataloader_source = (
        ROOT / "engine" / "data" / "dataloader.py"
    ).read_text(encoding="utf-8")
    assert "@register(name='SanitizeBoundingBoxes')" in transforms_source
    assert "@register(name='KeypointSanitizeBoundingBoxes')" in transforms_source
    assert "SanitizeBoundingBoxes = KeypointSanitizeBoundingBoxes" in transforms_source
    assert "'keypoints', 'keypoint_valid', 'has_keypoints'" in dataloader_source
    assert dataloader_source.count("updated_targets[i]['keypoints']") == 0
    assert "MixUp target schema mismatch" in dataloader_source


def test_named_registration_checks_the_registered_alias():
    workspace_source = (ROOT / "engine" / "core" / "workspace.py").read_text(
        encoding="utf-8")
    assert "assert register_name not in dct" in workspace_source
    assert "dct[register_name] = extract_schema(foo)" in workspace_source
    assert "assert foo.__name__ not in dct" not in workspace_source


def test_pose_sanitizer_preserves_bounding_box_metadata():
    source = (ROOT / "engine" / "data" / "transforms" / "_transforms.py").read_text(
        encoding="utf-8")
    assert "box_format = getattr(boxes, _boxes_keys[0]" in source
    assert "spatial_size = getattr(boxes, _boxes_keys[1]" in source
    assert "target['boxes'] = convert_to_tv_tensor(" in source


def test_training_validates_normalized_target_boxes():
    source = (ROOT / "engine" / "solver" / "ec_engine.py").read_text(
        encoding="utf-8")
    assert "_validate_target_boxes(targets)" in source
    assert "Target boxes must be normalized CXCYWH values in [0, 1]" in source
    assert "ConvertBoxes(fmt=\"cxcywh\", normalize=True)" in source


def test_training_gt_visualization_is_configurable_and_wired():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((
        ROOT / "configs" / "ecdetpose" / "examples"
        / "ecdetpose_s_vehicle_31kpts.yml"
    ).read_text(encoding="utf-8"))
    assert config["train_gt_visualization_interval"] == 10
    assert config["train_gt_visualization_images"] == 2

    engine_source = (ROOT / "engine" / "solver" / "ec_engine.py").read_text(
        encoding="utf-8")
    solver_source = (ROOT / "engine" / "solver" / "ec_solver.py").read_text(
        encoding="utf-8")
    base_config_source = (ROOT / "engine" / "core" / "_config.py").read_text(
        encoding="utf-8")
    assert "_visualize_training_ground_truth(" in engine_source
    assert "training_ground_truth" in engine_source
    assert "Training_ground_truth/sample_" in engine_source
    assert "train_gt_visualization_interval=getattr(" in solver_source
    assert "self.train_gt_visualization_interval :int = 0" in base_config_source


def test_decoupled_pose_head_optimizer_recipes():
    yaml = pytest.importorskip("yaml")
    examples = ROOT / "configs" / "ecdetpose" / "examples"
    for size, hidden in (("s", 384), ("m", 512)):
        config = yaml.safe_load((
            examples / f"ecdetpose_{size}_vehicle_decoupled_head.yml"
        ).read_text(encoding="utf-8"))
        transformer = config["ECTransformer"]
        assert transformer["keypoint_head_layers"] == 4
        assert transformer["keypoint_head_hidden_dim"] == hidden
        assert transformer["keypoint_visibility_head_layers"] == 3
        assert config["optimizer"]["type"] == "AdamW"
        assert config["optimizer"]["lr"] == pytest.approx(0.0005)
        assert any(group.get("lr") == pytest.approx(0.001)
                   for group in config["optimizer"]["params"])
        assert config["lr_gamma"] == pytest.approx(0.05)
        assert config["clip_max_norm"] == pytest.approx(0.1)


def test_optimizer_parameter_groups_reject_regex_overlap():
    source = (ROOT / "engine" / "core" / "yaml_config.py").read_text(
        encoding="utf-8")
    assert "Optimizer parameter regex" in source
    assert "overlaps a previous" in source
    engine_source = (ROOT / "engine" / "solver" / "ec_engine.py").read_text(
        encoding="utf-8")
    assert "group_name = pg.get('name', f'pg_{j}')" in engine_source


def test_visibility_head_depth_is_configurable():
    source = (ROOT / "engine" / "edgecrafter" / "decoder.py").read_text(
        encoding="utf-8")
    assert "keypoint_head_hidden_dim=None" in source
    assert "keypoint_visibility_head_layers=1" in source
    assert "final_layer = vis_head.layers[-1]" in source


def test_matcher_debug_logs_costs_matches_and_images():
    matcher_source = (ROOT / "engine" / "edgecrafter" / "matcher.py").read_text(
        encoding="utf-8")
    engine_source = (ROOT / "engine" / "solver" / "ec_engine.py").read_text(
        encoding="utf-8")
    solver_source = (ROOT / "engine" / "solver" / "ec_solver.py").read_text(
        encoding="utf-8")
    assert "return_diagnostics=False" in matcher_source
    assert "'weighted_components'" not in matcher_source  # implementation variable, not serialized state
    assert "result['diagnostics']" in matcher_source
    assert "_debug_hungarian_matches(" in engine_source
    assert "matches.jsonl" in engine_source
    assert "top_query_alternatives" in engine_source
    assert "normalized_keypoint_error_mean" in engine_source
    assert "Matcher_debug/sample_" in engine_source
    assert "matcher_debug_interval=getattr" in solver_source
