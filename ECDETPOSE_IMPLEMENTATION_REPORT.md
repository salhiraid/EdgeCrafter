# ECDetPose Implementation Report

## Summary

Implemented a trainable ECDetPose path that preserves ECDet detection components and adds decoder-query keypoint coordinate and visibility heads for configurable vehicle keypoints.

## Architectural Decisions

ECDetPose reuses ECDet `ViTAdapter`, `HybridEncoder`, and `ECTransformer`. Keypoints are predicted from the same decoder query features that produce class logits and boxes. `constrain_keypoints_to_box` decodes keypoints relative to each predicted box.

## Modified And Added Files

Modified:

`ecdetseg/engine/edgecrafter/modeling.py`
`ecdetseg/engine/edgecrafter/criterion.py`
`ecdetseg/engine/data/dataset/coco_dataset.py`
`ecdetseg/engine/data/transforms/_transforms.py`
`ecdetseg/engine/data/dataloader.py`

Added:

`ecdetseg/configs/ecdetpose/*.yml`
`ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_31kpts.yml`
`ecdetseg/tools/data/validate_coco_keypoints.py`
`ecdetseg/tools/checkpoint/convert_ecdet_to_ecdetpose.py`
`ecdetseg/tools/inference/ecdetpose_image.py`
`ecdetseg/tools/model_info.py`
`tests/*`
`docs/ecdetpose.md`
`docs/ecdetpose_model_zoo.md`

## Dataset Expectations

COCO JSON annotations must provide `[x, y, v] * 31` keypoints or omit keypoints for objects without keypoint labels. Malformed keypoint arrays raise errors with image and annotation IDs.

## Losses

Detection losses remain ECDet `loss_mal`, `loss_bbox`, `loss_giou`, `loss_fgl`, and `loss_ddf`. ECDetPose adds `loss_keypoint`, `loss_oks`, and `loss_keypoint_visibility`.

## Augmentation Behavior

Initial configs use keypoint-safe resize, optional configured horizontal flip, normalization, and box conversion. Mosaic and MixUp are disabled in ECDetPose configs pending complete keypoint-specific testing.

## Commands Executed

`git clone https://github.com/Intellindust-AI-Lab/EdgeCrafter.git EdgeCrafter`

`git switch -c feature/ecdetpose-vehicle-keypoints`

Pre-change smoke attempted:

`python -c "... YAMLConfig ... cfg.model ..."`

Result: failed because the active Python environment has no `torch` module installed.

Post-change checks:

`python -m py_compile ...`

Result: passed for changed Python modules, tools, and tests.

`python -c "... validate inline COCO keypoint JSON ..."`

Result: passed, reporting `2 31 1 0` for images, keypoints, empty images, malformed arrays.

`python -m pytest tests\test_coco_keypoint_validator.py`

Result: not run because `pytest` is not installed.

`python -c "... create_tiny_vehicle_keypoint_coco ..."`

Result: not run because `PIL` is not installed. The validator was patched so Pillow is required only for visualization mode.

`python -c "import yaml ..."`

Result: not run because `PyYAML` is not installed.

## Test Results

Passed:

`py_compile` syntax check for changed Python files.

Inline JSON-only COCO keypoint validator smoke.

Not run:

PyTorch model construction, forward/backward, AMP, training, resume, checkpoint conversion, and detection-equivalence tests because `torch` is not installed.

Pytest suite because `pytest` is not installed.

YAML config parse because `PyYAML` is not installed.

## Detection Equivalence

| Variant | Max Logit Diff | Max Bbox Diff | Passed |
|---|---:|---:|---|
| S | Not run | Not run | No |
| M | Not run | Not run | No |
| L | Not run | Not run | No |
| X | Not run | Not run | No |

## Model Size

Not measured. Run `python ecdetseg/tools/model_info.py --configs ...` in a PyTorch environment.

## Checkpoints

No ECDet source `.pth` checkpoints were present, and PyTorch is unavailable. Therefore no ECDetPose initialization checkpoints were generated.

## Visualization Paths

No inference visualizations were generated because checkpoints, test images, and PyTorch are unavailable.

## Known Limitations

Real-data smoke testing was not run because dataset paths in the request are placeholders.

S/M/L/X checkpoint conversion was not run because source ECDet checkpoints were not available.

Mosaic, affine, crop, and MixUp are disabled in initial ECDetPose configs.

## Recommended First Training Command

```bash
python ecdetseg/train.py \
  -c ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_31kpts.yml \
  -t checkpoints/ecdetpose_s_init_from_ecdet.pth \
  --use-amp
```

## Debugging Keypoint Loss

Validate the COCO file first, confirm `num_keypoints`, `keypoint_flip_pairs`, and `keypoint_oks_sigmas` lengths, train without horizontal flip until loss decreases, then inspect `loss_keypoint`, `loss_oks`, `loss_keypoint_visibility`, and keypoint-head gradient norms.
