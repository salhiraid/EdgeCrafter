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
