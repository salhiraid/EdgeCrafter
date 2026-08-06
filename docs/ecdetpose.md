# ECDetPose

ECDetPose extends ECDet with a vehicle keypoint branch that predicts class logits, boxes, keypoint coordinates, and keypoint confidence for the same decoder object query.

## Architecture

ECDetPose preserves the ECDet backbone, hybrid encoder, transformer decoder, classification heads, and box heads. The ECTransformer decoder optionally adds lightweight per-decoder-layer keypoint heads:

```python
{
    "pred_logits": [B, Q, C],
    "pred_boxes": [B, Q, 4],
    "pred_keypoints": [B, Q, K, 2],
    "pred_keypoint_logits": [B, Q, K],
}
```

`K` is configurable with `num_keypoints` and defaults to `31`.

When `constrain_keypoints_to_box: true`, raw keypoint predictions are decoded through a bbox-relative sigmoid parameterization, so coordinates remain inside the associated predicted box:

```python
relative_xy = sigmoid(raw_xy)
x1 = cx - 0.5 * w
y1 = cy - 0.5 * h
x = x1 + relative_xy[..., 0] * w
y = y1 + relative_xy[..., 1] * h
```

## Data Format

Training uses COCO annotations with per-object keypoints:

```json
{
  "bbox": [x, y, width, height],
  "category_id": 1,
  "keypoints": [x1, y1, v1, "..."],
  "num_keypoints": 31
}
```

Visibility follows COCO semantics:

`0`: not labeled; ignored by coordinate and OKS losses, with confidence target 0 for keypoint-annotated instances.

`1`: labeled but not visible, supervised for coordinates and confidence target 0.

`2`: labeled and visible, supervised for coordinates and confidence target 1.

The loader validates `len(keypoints) == 3 * num_keypoints` and reports image/annotation IDs for malformed objects.

Mixed bbox/keypoint datasets are supported. If an annotation omits `keypoints` (or supplies an empty array), the loader inserts a `[K, 3]` zero placeholder and sets its per-instance `keypoint_valid` flag to false. This preserves one-to-one instance alignment through filtering and Mosaic-style concatenation, while the matcher and all three keypoint losses ignore that bbox's placeholder. Detection classification and box losses still use the bbox normally. An annotation that contains a correctly sized all-`v=0` keypoint array is different: it is an annotated pose, so it supervises confidence as absent but has no coordinate or OKS contribution.

## Augmentations

Initial ECDetPose configs use keypoint-safe resize, optional horizontal flip with configured flip pairs, normalization, and box conversion. Mosaic and MixUp are disabled by default because this implementation does not claim full tested keypoint support for those paths.

Horizontal flip is disabled with a warning if `keypoint_flip_pairs` is empty and `allow_without_flip_pairs` is false.

## Losses And Matching

Hungarian matching remains detection-driven by default:

```yaml
keypoint_cost_weight: 0.0
oks_cost_weight: 0.0
```

After detection matching, ECDetPose adds:

`loss_keypoint`: visibility-masked Smooth L1 coordinate loss.

`loss_oks`: configurable-sigma OKS loss.

`loss_keypoint_visibility`: visibility/confidence BCE loss over all joints of keypoint-annotated instances only.

Do not use 17-person-keypoint COCO sigmas for 31 vehicle keypoints. Set `keypoint_oks_sigmas` to 31 values.

## Checkpoint Conversion

```bash
python ecdetseg/tools/checkpoint/convert_ecdet_to_ecdetpose.py \
  --ecdet-config ecdetseg/configs/ecdet/ecdet_s.yml \
  --ecdetpose-config ecdetseg/configs/ecdetpose/ecdetpose_s.yml \
  --ecdet-checkpoint checkpoints/ecdet_s.pth \
  --num-keypoints 31 \
  --output checkpoints/ecdetpose_s_init_from_ecdet.pth
```

The converter refuses unexpected detection-weight misses. Only keypoint-head parameters may be missing from the source ECDet checkpoint.

## Dataset Validation

```bash
python ecdetseg/tools/data/validate_coco_keypoints.py \
  --ann-file <PATH_TO_TRAIN_COCO_KEYPOINT_JSON> \
  --num-keypoints 31 \
  --image-root <PATH_TO_TRAIN_IMAGES> \
  --visualize-output outputs/ecdetpose_dataset_samples
```

## Training

```bash
python ecdetseg/train.py \
  -c ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_31kpts.yml \
  -t checkpoints/ecdetpose_s_init_from_ecdet.pth \
  --use-amp
```

Resume from a full training checkpoint:

```bash
python ecdetseg/train.py \
  -c ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_31kpts.yml \
  -r outputs/ecdetpose_s_vehicle_31kpts/last.pth \
  --use-amp
```

## Inference

```bash
python ecdetseg/tools/inference/ecdetpose_image.py \
  --config ecdetseg/configs/ecdetpose/ecdetpose_s.yml \
  --checkpoint checkpoints/ecdetpose_s_init_from_ecdet.pth \
  --input <PATH_TO_TEST_IMAGES> \
  --output-dir outputs/ecdetpose_inference \
  --device cuda \
  --score-threshold 0.4 \
  --keypoint-threshold 0.0 \
  --save-json \
  --show-labels \
  --show-keypoint-indices
```

Initial converted checkpoints have meaningful pretrained boxes but random keypoints. Visualizations and logs explicitly warn that the keypoint head is randomly initialized.

## Known Limitations

Mosaic, random affine, random crop, and MixUp are not enabled in the initial ECDetPose configs. Enable them only after adding numeric keypoint transform tests for the exact policy.

Real S/M/L/X initialization checkpoints require the corresponding ECDet checkpoints locally. They were not generated by this documentation file.

## Training From Multiple Datasets

Use `ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_5datasets.yml` to train from five COCO datasets. Replace all ten path placeholders and adjust:

```yaml
weights: [0.35, 0.25, 0.20, 0.12, 0.08]
samples_per_epoch: 50000
```

The weights are **dataset-level sampling probabilities**, not per-image weights. They are normalized automatically: with the values above, approximately 35% of an epoch comes from dataset 1 even if dataset 1 is much smaller or larger than the others. Sampling is with replacement, which is appropriate when intentionally oversampling a small dataset. `samples_per_epoch` controls the epoch length and therefore the learning-rate schedule cadence.

`WeightedMultiDataset` maps every virtual sample index deterministically from `seed`, epoch, and index. Single-process training uses the normal shuffled loader. Distributed training continues to use the project's `DistributedSampler`, which shards those virtual indices across ranks while preserving the same global weighted mixture. All component datasets must use the same category-label mapping and the same 31-keypoint order. Bbox-only annotations are allowed because the loader supplies ignored keypoint placeholders.

## COCO Evaluation Metrics

The vehicle example evaluates both detection and pose on the single dataset configured under `val_dataloader`:

```yaml
evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', 'keypoints']
  verbose: true
  keypoint_oks_sigmas: [0.025, ...]  # exactly 31 values
```

During training, validation runs after every epoch and prints the standard COCO summaries:

- `bbox`: AP, AP50, AP75, AP-small, AP-medium, AP-large, and the corresponding AR metrics.
- `keypoints`: OKS AP, AP50, AP75, AP-medium, AP-large, AR, AR50, AR75, AR-medium, and AR-large.

The same arrays are appended to `outputs/ecdetpose_s_vehicle_31kpts/log.txt` as `test_coco_eval_bbox` and `test_coco_eval_keypoints`. To evaluate a checkpoint without training, run:

```bash
python ecdetseg/train.py \
  -c ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_5datasets.yml \
  -r <PATH_TO_FULL_CHECKPOINT> \
  --test-only
```

The five weighted datasets are training sources only. COCO metrics are calculated against the one non-weighted COCO validation dataset inherited from `ecdetpose_s_vehicle_31kpts.yml`; set its `val_dataloader.dataset.img_folder` and `ann_file` to the validation set you want to report. This avoids mixing image IDs and incompatible COCO ground-truth objects. To report each of five validation datasets separately, run evaluation five times with a different validation path override/config for each run.

For keypoint evaluation, every category in the validation JSON should define the same 31 keypoint names, and annotated arrays must use that order. Bbox-only validation annotations are automatically converted to ignored, zero-keypoint ground truths and therefore affect bbox metrics but not OKS metrics. Predictions are exported to pycocotools as the required `(x, y, v)` triplets, while the configured 31-value sigma vector replaces pycocotools' incompatible 17-person-keypoint default.
