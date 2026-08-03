# ECDetPose Implementation Report

## Summary

Implemented and smoke-tested ECDetPose as an ECDet-preserving multi-task path with same-query classification, bbox detection, keypoint coordinates, and keypoint visibility logits.

## Architectural Decisions

ECDetPose reuses ECDet `ViTAdapter`, `HybridEncoder`, and `ECTransformer`. Keypoints are predicted from the same decoder query features that produce class logits and boxes. `constrain_keypoints_to_box` decodes keypoints relative to each predicted box, keeping constrained predictions inside the associated box.

## Modified And Added Files

Key implementation files:

- `ecdetseg/engine/edgecrafter/modeling.py`
- `ecdetseg/engine/edgecrafter/decoder.py`
- `ecdetseg/engine/edgecrafter/criterion.py`
- `ecdetseg/engine/edgecrafter/matcher.py`
- `ecdetseg/engine/edgecrafter/postprocessor.py`
- `ecdetseg/engine/data/dataset/coco_dataset.py`
- `ecdetseg/engine/data/transforms/_transforms.py`
- `ecdetseg/engine/data/dataloader.py`
- `ecdetseg/tools/data/validate_coco_keypoints.py`
- `ecdetseg/tools/checkpoint/convert_ecdet_to_ecdetpose.py`
- `ecdetseg/tools/inference/ecdetpose_image.py`
- `ecdetseg/tools/model_info.py`
- `ecdetseg/configs/ecdetpose/*.yml`

## Dataset Expectations

COCO JSON annotations must provide `[x, y, v] * K` keypoints or omit keypoints for objects without keypoint labels. Malformed keypoint arrays raise errors with image and annotation IDs. Smoke testing used a tiny local human-pose-style COCO sample with 17 keypoints.

## Losses

Detection losses remain ECDet `loss_mal`, `loss_bbox`, `loss_giou`, `loss_fgl`, and `loss_ddf`. ECDetPose adds visibility-masked `loss_keypoint`, configurable-sigma `loss_oks`, and `loss_keypoint_visibility`.

## Augmentation Behavior

Initial configs use keypoint-safe resize, optional configured horizontal flip, normalization, and box/keypoint conversion. Mosaic and MixUp are disabled in ECDetPose configs pending complete keypoint-specific testing.

## Commands Executed

- Created `.venv-ecdetpose`.
- Installed CPU runtime: `torch==2.3.1+cpu`, `torchvision==0.18.1+cpu`, Pillow, pytest, PyYAML, SciPy, pycocotools, tensorboard, calflops, transformers, einops.
- `python -m compileall ecdetseg\engine\edgecrafter ecdetseg\engine\data ecdetseg\tools`
- `pytest ecdetseg/tests -q`
- Generated random-source ECDet-S checkpoint.
- Converted ECDet-S random-source checkpoint to ECDetPose-S with the converter CLI.
- Ran ECDet vs ECDetPose-S detection-equivalence smoke on a dummy tensor.
- Ran `ecdetpose_image.py` on a local smoke image.
- Ran one real ECDetPose-S optimizer step on a tiny 17-keypoint human-pose-style sample.
- Ran `tools/model_info.py` for ECDetPose-S.

## Test Results

Passed:

- Syntax compilation for changed engine/data/tools files.
- `pytest ecdetseg/tests -q`: `3 passed, 1 skipped`.
- ECDetPose-S raw output format:
  `pred_logits [1, 300, 80]`, `pred_boxes [1, 300, 4]`, `pred_keypoints [1, 300, 31, 2]`, `pred_keypoint_logits [1, 300, 31]`.
- ECDet vs ECDetPose-S detection equivalence from the converted checkpoint:
  max logit diff `0.0`, max bbox diff `0.0`.
- Inference output serialization: raw predictions, COCO bbox predictions, COCO keypoint predictions, timing JSON, and visualization PNG.
- Constrained prediction check: `100.00%` keypoints inside associated boxes.
- Human-pose-style 17-keypoint training smoke:
  finite total loss `63.1905`; optimizer step completed.

Gradient norms from the training smoke:

| Component | Grad Norm |
| --- | ---: |
| backbone | 96.0737 |
| encoder | 25.1307 |
| decoder | 2.8018 |
| classification head | 1.2325 |
| bbox head | 2.1304 |
| keypoint coordinate head | 0.4212 |
| keypoint visibility head | 11.9883 |

Not run:

- M/L/X runtime conversion/equivalence. Official pretrained ECDet M/L/X checkpoints were not present.
- Full overfit loop, CUDA AMP, and resume. The machine run used CPU; a real optimizer step was executed.
- Real private dataset smoke, because the dataset path was not provided.

## Detection Equivalence

| Variant | Max Logit Diff | Max Bbox Diff | Passed |
| --- | ---: | ---: | --- |
| S | 0.0 | 0.0 | Yes |
| M | Not run | Not run | No |
| L | Not run | Not run | No |
| X | Not run | Not run | No |

## Model Size

Measured for ECDetPose-S:

| Variant | Params | MACs | FLOPs | Input | Queries | Decoder Layers |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| S | 10.25M | 13.05 GMACs | 26.21 GFLOPs | 640x640 | 300 | 4 |

## Checkpoints

Generated smoke checkpoints are random-source initialization tests, not official pretrained detection checkpoints:

- `checkpoints/ecdet_s_random_source.pth`
  SHA-256: `9B67F93576FBF01FC7A8C33FEA0E99CC87A2DCDC1952387EE0C9180AD62B566A`
- `checkpoints/ecdetpose_s_init_from_random_ecdet.pth`
  SHA-256: `AFB7CF38D9177C247168CF858CB45EEA83CDD5A210E2E27FC304B1BC075E9108`
- `checkpoints/ecdetpose_s_cli_init_from_random_ecdet.pth`
  SHA-256: `C28C035AC8441D03CD4D68CA4F43DE6A8A1499BE1DB8400D43FD21169BC18BF5`

Expected official outputs once official ECDet checkpoints are supplied:

- `checkpoints/ecdetpose_s_init_from_ecdet.pth`
- `checkpoints/ecdetpose_m_init_from_ecdet.pth`
- `checkpoints/ecdetpose_l_init_from_ecdet.pth`
- `checkpoints/ecdetpose_x_init_from_ecdet.pth`

## Visualization Paths

- `outputs/ecdetpose_smoke/inference/smoke_vehicle_like.png`
- `outputs/ecdetpose_smoke/inference/raw_predictions.json`
- `outputs/ecdetpose_smoke/inference/coco_bbox_predictions.json`
- `outputs/ecdetpose_smoke/inference/coco_keypoint_predictions.json`
- `outputs/ecdetpose_smoke/inference/timing.json`
- `outputs/ecdetpose_human_smoke/single_step_training_report.json`

## Known Limitations

The generated S checkpoints are random-source smoke artifacts, not official pretrained ECDet detection weights. M/L/X runtime validation remains to be run with their source checkpoints. Mosaic, affine, rotation, and MixUp keypoint support remain disabled pending dedicated geometric tests. Push was not performed because `origin` points to the upstream public repository, not a user fork.

## Recommended First Training Command

```bash
python ecdetseg/train.py \
  -c ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_31kpts.yml \
  --pretrained checkpoints/ecdetpose_s_init_from_ecdet.pth \
  --amp
```

## Debugging Keypoint Loss

Validate the COCO file first, confirm `num_keypoints`, `keypoint_flip_pairs`, and `keypoint_oks_sigmas` lengths, train without horizontal flip until loss decreases, then inspect `loss_keypoint`, `loss_oks`, `loss_keypoint_visibility`, and keypoint-head gradient norms.
