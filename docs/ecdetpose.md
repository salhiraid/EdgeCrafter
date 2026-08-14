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

The five-dataset config changes the inherited dataset `type` from
`CocoDetection` to `WeightedMultiDataset`. Config merging treats that type
change as a complete node replacement, so inherited `img_folder`, `ann_file`,
and `num_keypoints` fields are not passed to `WeightedMultiDataset`. If you see
`unexpected keyword argument 'img_folder'`, update `engine/core/yaml_utils.py`
and the five-dataset config from the same revision.

`WeightedMultiDataset` exposes the shared composed augmentation pipeline under
both `transforms` and the legacy `_transforms` attribute expected by
`ECSolver`. The solver also accepts either name when reading `stop_epoch` and
`mosaic_epoch`. This compatibility is required before the first training epoch;
if `_transforms` is missing, update `engine/data/dataset/_dataset.py` and
`engine/solver/ec_solver.py` together.

The wrapper also remaps every component COCO `category_id` to one shared,
contiguous label space by category name. By default, the category order in the
first JSON becomes labels `0..C-1`; all other JSON files must contain the same
category-name set. Set `category_names` on `WeightedMultiDataset` to make the
order explicit. Its length must equal `num_classes`. This prevents raw COCO ids
such as `7` from reaching a five-class denoising/gather kernel, which otherwise
causes a delayed CUDA `vectorized_gather_kernel index out of bounds` assertion.
The training loop additionally validates every target label before model
forward and reports the dataset index and image id in a synchronous Python
error if a label is outside `[0, num_classes - 1]`.

## COCO Evaluation Metrics

The vehicle example evaluates both detection and pose on the single dataset configured under `val_dataloader`:

```yaml
evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', 'keypoints']
  verbose: true
  keypoint_score_mode: bbox_keypoint
  keypoint_score_thr: 0.2
  keypoint_distance_thresholds: [5, 10]
  keypoint_visibility_thr: 0.5
  keypoint_match_iou_thr: 0.5
  pose_detection_score_thr: 0.3
  pose_crop_size: 512
  pose_crop_margin: 0.05
  pose_min_bbox_size: 128
  keypoint_oks_sigmas: [0.025, ...]  # exactly 31 values
```

During training, validation runs after every epoch and prints the standard COCO summaries:

- `bbox`: AP, AP50, AP75, AP-small, AP-medium, AP-large, and the corresponding AR metrics.
- `keypoints`: OKS AP, AP50, AP75, AP-medium, AP-large, AR, AR50, AR75, AR-medium, and AR-large.

The same arrays are appended to `outputs/ecdetpose_s_vehicle_31kpts/log.txt` as `test_coco_eval_bbox` and `test_coco_eval_keypoints`. TensorBoard records every value with a readable name rather than a numeric suffix:

```text
Performance/BBox/AP
Performance/BBox/AP50
Performance/BBox/AP75
Performance/BBox/AP_small
Performance/BBox/AP_medium
Performance/BBox/AP_large
Performance/Keypoints_COCO_OKS/AP
Performance/Keypoints_COCO_OKS/AP50
Performance/Keypoints_COCO_OKS/AP75
Performance/Keypoints_COCO_OKS/AP_medium
Performance/Keypoints_COCO_OKS/AP_large
Performance/Keypoints_Pixel/Precision_5px
Performance/Keypoints_Pixel/Recall_5px
Performance/Keypoints_Pixel/F1_5px
Performance/Keypoints_Pixel/Precision_10px
Performance/Keypoints_Pixel/Recall_10px
Performance/Keypoints_Pixel/F1_10px
Performance/Keypoints_Pixel/Visibility_Precision
Performance/Keypoints_Pixel/Visibility_Recall
Performance/Keypoints_Pixel/Visibility_F1
Performance/Keypoints_Pixel/Visibility_Accuracy
Performance/Keypoints_Pixel/Matched_Instances
Performance/Keypoints_Pixel/Eligible_GT_Instances
```

TensorBoard therefore shows bbox performance, COCO-OKS keypoint performance,
and pixel-distance keypoint performance as three independent groups. They are
not averaged or merged together.

The remaining named AR metrics are logged in the same groups. Start TensorBoard with `tensorboard --logdir outputs/ecdetpose_s_vehicle_31kpts/summary` (or the configured output directory's `summary` folder).

If evaluation raises `KeyError: 'pose'` from `metric_names`, the runtime is
mixing an older `coco_eval.py` with the newer solver. Update both files from
the same revision. The default `pose` names are also registered directly in
`COCO_METRIC_NAMES` so the 5 px and 10 px metrics remain compatible with that
older name-lookup path.

For pose ranking, `bbox_keypoint` follows the MMPose strategy: it multiplies the detection score by the mean confidence of keypoints above `keypoint_score_thr`. Set `keypoint_score_mode: bbox` to reproduce bbox-only ranking, or `keypoint` to rank only by mean keypoint confidence. DETR predictions remain NMS-free by design; no additional OKS NMS is applied.

The pixel-distance metrics mirror the supplied MMDetection evaluation. They
first retain detections with score at least `pose_detection_score_thr`, then
greedily match predictions to same-category ground truths at
`keypoint_match_iou_thr`. Instances smaller than `pose_min_bbox_size` are
excluded. Each matched bbox is expanded by `pose_crop_margin`, projected into
an aspect-ratio-preserving `pose_crop_size × pose_crop_size` crop, and the 5 px
and 10 px distances are measured in that crop space. Only visible ground-truth
joints (`v=2`) enter coordinate precision/recall/F1. Visibility metrics treat
`v=2` as visible and `v=0/1` as not visible. Bbox-only annotations are excluded.
The aggregate curves are macro averages across keypoint indices, matching the
reference implementation rather than pooling every joint into one micro count.

TensorBoard also exposes every joint separately under paths such as:

```text
Performance/Keypoints_PerJoint/front_light_left/Precision_5px
Performance/Keypoints_PerJoint/front_light_left/Recall_10px
Performance/Keypoints_PerJoint/front_light_left/Visibility_Accuracy
```

Every validation pass also renders up to 10 images with green predicted boxes.
The renderer resolves each `image_id` through the validation COCO metadata and
loads `val_dataloader.dataset.img_folder / file_name`, so both the saved JPEG
and the TensorBoard image use the **original image resolution**. It falls back
to the transformed model input only when the original file cannot be found.
All predicted keypoints belonging to retained boxes are drawn: red means the
visibility score passed `keypoint_visibility_thr`, while orange means the
keypoint branch produced a coordinate but its visibility score is below the
threshold. Keypoint indices are printed next to the points. This makes a weak
or newly initialized visibility head visible during debugging instead of
producing apparently empty images. The images appear in TensorBoard under
`Validation_predictions` and are saved as JPEGs under:

```text
<output_dir>/prediction_visualizations/epoch_XXXX/image_<image_id>.jpg
```

Named evaluation scalars are written during both epoch validation and
`--test-only` validation. If no boxes or keypoints appear in the images, lower
`pose_detection_score_thr`; this threshold selects the object queries rendered
and included in the pixel-distance metrics.

To evaluate a checkpoint without training, run:

```bash
python ecdetseg/train.py \
  -c ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_5datasets.yml \
  -r <PATH_TO_FULL_CHECKPOINT> \
  --test-only
```

The five weighted datasets are training sources only. COCO metrics are calculated against the one non-weighted COCO validation dataset inherited from `ecdetpose_s_vehicle_31kpts.yml`; set its `val_dataloader.dataset.img_folder` and `ann_file` to the validation set you want to report. This avoids mixing image IDs and incompatible COCO ground-truth objects. To report each of five validation datasets separately, run evaluation five times with a different validation path override/config for each run.

For keypoint evaluation, every category in the validation JSON should define the same 31 keypoint names, and annotated arrays must use that order. Bbox-only validation annotations are automatically converted to ignored, zero-keypoint ground truths and therefore affect bbox metrics but not OKS metrics. Predictions are exported to pycocotools as the required `(x, y, v)` triplets, while the configured 31-value sigma vector replaces pycocotools' incompatible 17-person-keypoint default.
### Target-size mismatch in the Hungarian matcher

Pose pipelines must use `KeypointSanitizeBoundingBoxes`, not torchvision's
generic `SanitizeBoundingBoxes`. The pose-aware sanitizer applies the same keep
mask to boxes, labels, keypoints, `keypoint_valid`, areas, masks, and crowd
flags. Using the generic sanitizer can leave more keypoint rows than boxes and
produce a matcher error such as `tensor a (...) must match tensor b (...)`.

## Pose-accuracy training recipes

Six vehicle recipes are provided for the S and M models:

| Goal | S config | M config |
|---|---|---|
| Balanced first training | `ecdetpose_s_vehicle_pose_balanced.yml` | `ecdetpose_m_vehicle_pose_balanced.yml` |
| Maximum spatial precision | `ecdetpose_s_vehicle_pose_precision.yml` | `ecdetpose_m_vehicle_pose_precision.yml` |
| Low-LR second-stage tuning | `ecdetpose_s_vehicle_pose_finetune.yml` | `ecdetpose_m_vehicle_pose_finetune.yml` |

Start with **balanced**. It raises coordinate/OKS loss weights and enables an
L1 keypoint cost in Hungarian matching without removing the detection losses.
Use **precision** when small or distant keypoints need more pixels: it trains and
evaluates at 960×960 and disables Mosaic/MixUp, but needs more GPU memory. Use
**finetune** only as a second stage from a good full pose checkpoint; it uses a
10× smaller head learning rate, a 10× smaller backbone learning rate, and no
Mosaic/MixUp. Do not resume finetuning from a bbox-only checkpoint whose
keypoint head was never trained.

Example balanced training:

```bash
python ecdetseg/train.py \
  -c ecdetseg/configs/ecdetpose/examples/ecdetpose_s_vehicle_pose_balanced.yml \
  --use-amp
```

Example second-stage M-model fine-tuning:

```bash
python ecdetseg/train.py \
  -c ecdetseg/configs/ecdetpose/examples/ecdetpose_m_vehicle_pose_finetune.yml \
  -t outputs/ecdetpose_m_vehicle_pose_balanced/best.pth \
  --use-amp
```

Replace the inherited train/validation path placeholders before training. For a
`WeightedMultiDataset`, copy the recipe's `ECCriterion`, matcher, resolution,
and augmentation overrides into the five-dataset config rather than replacing
its outer dataset node.

When comparing recipes, monitor all three separate groups:
`Performance/BBox`, `Performance/Keypoints_COCO_OKS`, and
`Performance/Keypoints_Pixel`. If bbox AP falls while pose improves, reduce
`loss_keypoint` or `keypoint_cost_weight`; if coordinates improve but visibility
is poor, raise `loss_keypoint_visibility` gradually. Also audit annotation order,
visibility values, bbox-only rates per source dataset, and per-joint metrics—the
optimizer cannot correct inconsistent keypoint semantics across datasets.

### Loss and matcher ablation recipes

Three additional S/M pairs isolate different pose bottlenecks:

| Recipe suffix | Loss weights `(coordinate, OKS, visibility)` | Matcher `(keypoint, OKS)` | Use when |
|---|---:|---:|---|
| `pose_coordinate` | `(20, 2, 1)` | `(6, 0)` | Coordinates are consistently displaced but visibility is acceptable |
| `pose_oks` | `(8, 10, 1)` | `(1, 4)` | Scale-normalized COCO OKS is the primary target |
| `pose_visibility` | `(10, 4, 3)` | `(2, 0)` | Visibility recall/F1 is the main failure |

Each suffix is available as both `ecdetpose_s_vehicle_<suffix>.yml` and
`ecdetpose_m_vehicle_<suffix>.yml`. Change one axis at a time and compare it to
the balanced recipe using the same seed, data mixture, batch size, and
checkpoint initialization. The OKS matcher uses the same 31 sigmas as the
criterion and normalized bbox area; bbox-only targets receive zero pose cost.
Avoid choosing the largest weight merely because it is available: overly strong
pose matching can reduce bbox AP or destabilize early assignments.

### Alignment error after enabling keypoint matcher costs

An error such as `boxes has 3 instances but keypoints has 5` is a data-pipeline
alignment failure, not a consequence of the numerical loss weights. It can be
revealed when `keypoint_cost_weight` or `oks_cost_weight` changes from zero,
because the matcher then reads the previously unused keypoint matrix.

Both sanitizer names now resolve to the pose-aware sanitizer, which applies one
keep mask to every instance field. MixUp likewise concatenates boxes, labels,
areas, crowd flags, masks, keypoints, and validity flags exactly once and rejects
incompatible schemas. Update both `_transforms.py` and `dataloader.py` on older
training checkouts. Retain `KeypointSanitizeBoundingBoxes` explicitly in new pose
configs because it documents the required behavior.

### Abnormally large bbox loss after pose sanitization

The criterion expects normalized `CXCYWH` boxes, so a weighted `loss_bbox` in
the thousands means absolute pixel `XYXY` boxes reached the loss. On some
Torchvision versions, boolean indexing in the pose-aware sanitizer drops the
`BoundingBoxes` subclass and its format/canvas metadata. `ConvertBoxes` then
skips that plain tensor because it only transforms `BoundingBoxes` values.

The sanitizer now explicitly reconstructs the `BoundingBoxes` value after
filtering. Training also validates the normalized `[0, 1]` target-box contract
before the model runs and reports dataset/image context. With `loss_bbox: 5`, a
properly normalized per-layer bbox loss should be on the order of units, not
thousands. The total displayed loss sums the main, auxiliary, encoder, and
denoising losses, so one malformed bbox loss is repeated many times.

### Inspect training augmentation with GT overlays

Set `train_gt_visualization_interval: 10` to render random pose-annotated samples from the
post-augmentation training batch (falling back to any sample if none has pose GT) every ten global iterations. Set
`train_gt_visualization_images` to the number of random batch items saved at
each interval. Images are denormalized, while normalized `CXCYWH` boxes and
normalized keypoints are projected back onto the transformed model input.
Green rectangles are GT boxes, red points are visible (`v=2`) keypoints, orange
points are labeled-but-occluded (`v=1`) keypoints, and `v=0` or bbox-only
placeholders are omitted.

JPEGs are written to
`<output_dir>/training_ground_truth/epoch_XXXX/step_XXXXXXXX_image_<id>.jpg` and
the same images appear in TensorBoard under `Training_ground_truth/sample_N`.
These are intentionally transformed training images—not original-resolution
files—so they show exactly what the model receives after resize, flip, MixUp,
and other enabled augmentation. Set the interval to `0` to disable this output.

## Optimizer, learning-rate, and decoupled pose-head recipe

ECDetPose uses regex-based optimizer parameter groups. Parameters matched by a
group receive that group's `lr` and `weight_decay`; unmatched trainable
parameters receive the optimizer-level defaults. Regexes must not overlap.
`FlatCosineLRScheduler` applies quadratic iteration warmup, cosine decay to
`base_lr * lr_gamma`, then a constant no-augmentation tail.

The `ecdetpose_{s,m}_vehicle_decoupled_head.yml` recipes provide a
YOLOX-Pose-inspired separation while retaining the transformer query design:
a four-layer coordinate MLP and a separate three-layer visibility MLP are
attached to each decoder layer. The S recipe uses 384 hidden channels and the M
recipe uses 512. This is not a convolutional YOLOX head or SimOTA—the existing
Hungarian query assignment remains—but coordinate and visibility branches have
independent capacity.

The recipes use AdamW with default non-backbone LR `5e-4`, backbone LR `2.5e-5`,
and pose-head LR `1e-3`. Norm/bias decay is zero, other decay is `1e-4`, warmup
is 2000 iterations, minimum LR is 5% of each parameter group's base LR, gradient
clipping is `0.1`, and EMA is enabled. Start with these values rather than YOLOX
SGD `0.005`: transformer training and effective batch sizes differ materially.
Scale all three learning rates together when changing global batch size, then
adjust the head multiplier separately only if pose learns too slowly or becomes
unstable.

Named optimizer groups are logged as `Lr/backbone`,
`Lr/backbone_no_decay`, `Lr/pose_head`, `Lr/pose_head_no_decay`, and
`Lr/other_no_decay`; unmatched parameters appear in the automatically appended
default group. To fine-tune from a checkpoint while applying these new optimizer
settings, use `-t`; `-r` restores the checkpoint optimizer/scheduler state and
is intended only for continuing the same run.

### Hungarian matching for bbox-only ground truth

A missing or empty COCO `keypoints` array is loaded as a zero placeholder with
`keypoint_valid=False`. The matcher builds `pose_valid` from both that instance
flag and per-joint `v>0`. It allocates zero keypoint/OKS cost matrices, computes
pose distances only for `pose_valid` target columns, and scatters those costs
back into the full matrix. Consequently a bbox-only target is assigned using
only classification, bbox L1, and GIoU costs even when
`keypoint_cost_weight`/`oks_cost_weight` are nonzero. An annotated instance with
all joints at `v=0` is also excluded from coordinate/OKS matching, while its
visibility labels can still be supervised by the criterion.

### Inspect Hungarian assignments

Set `matcher_debug_interval` and `matcher_debug_images` at the config root to
periodically rerun the main-output matcher in diagnostic mode. It writes
`<output_dir>/matcher_debug/matches.jsonl`, one JSON object per matched pair,
including image/target/query ids, GT label, predicted label and class score,
pose-valid flags, total cost, weighted class/bbox/GIoU/keypoint/OKS cost terms,
normalized GT/pred boxes, five lowest-cost alternative queries, mean/max
keypoint error, and mean predicted visibility where applicable.

Matched overlays are saved under
`matcher_debug/epoch_XXXX/step_XXXXXXXX_image_<id>.jpg` and TensorBoard
`Matcher_debug/sample_N`. Green denotes GT, blue boxes denote matched
predictions, red points denote matched predictions, green points denote GT, and
yellow segments show joint error. Bbox-only targets have no pose points and
must show zero `keypoint`/`oks` weighted costs in JSON. Use a moderate interval
such as 100 because diagnostic matching adds computation and disk I/O.
## Dataset inference and sequential prediction images

Use `ecdetpose_dataset.py` to run a checkpoint over the exact image list and
image ids in a COCO annotation file. It writes combined results, COCO bbox and
keypoint result files, and optionally prediction overlays named `00001.jpg`,
`00002.jpg`, and so on:

```bash
CUDA_VISIBLE_DEVICES=0 python ecdetseg/tools/inference/ecdetpose_dataset.py \
  --config ecdetseg/configs/ecdetpose/examples/ecdetpose_m_vehicle_31kpts.yml \
  --checkpoint /path/to/best.pth \
  --ann-file /data/vehicle/annotations/val.json \
  --image-root /data/vehicle/images \
  --save-dir outputs/vehicle_val_predictions \
  --inference-size 960x960 \
  --visualization-resolution original \
  --score-threshold 0.3 \
  --keypoint-threshold 0.5 \
  --category-ids 1,2,3,4,5 \
  --show-labels \
  --show-keypoint-indices \
  --amp
```

`--inference-size` is `HEIGHTxWIDTH`. If it is omitted, the script uses the
configuration's `eval_spatial_size`. `--visualization-resolution original`
draws on the source image at its original dimensions. Use
`--visualization-resolution inference` to save the resized model input instead.

The model predicts normalized boxes and keypoints. The postprocessor scales X
by the selected output width and Y by the selected output height. Consequently,
even when inference resizing changes the aspect ratio, JSON coordinates are
always converted back to the original-image coordinate system; inference-size
overlays use a second, independent projection onto the resized image. Boxes and
keypoints therefore remain aligned in both output modes.

The `--category-ids` values must be COCO category ids in model-label order. For
example, label `0` maps to the first id, label `1` to the second id, and so on.
If omitted, category order from the annotation JSON is used.

The output directory contains:

```text
vehicle_val_predictions/
├── images/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── predictions.json
├── coco_bbox_predictions.json
├── coco_keypoint_predictions.json
└── summary.json
```

`predictions.json` retains the source file name, original/inference sizes, and
all detections per image. The two `coco_*` files can be passed directly to COCO
evaluation. Add `--no-images` for JSON-only inference, `--max-images 100` for a
short run, or `--start-index 0 --filename-digits 6` to produce `000000.jpg`.

## Export ECDetPose to ONNX

Export a pose checkpoint with:

```bash
python ecdetseg/tools/deployment/export_onnx.py \
  --config ecdetseg/configs/ecdetpose/examples/ecdetpose_m_vehicle_31kpts.yml \
  --resume /path/to/best.pth \
  --opset 18 \
  --check
```

To export a graph whose input is **960 pixels wide and 544 pixels high**, pass
the CLI dimensions in `HEIGHT WIDTH` order:

```bash
python ecdetseg/tools/deployment/export_onnx.py \
  --config ecdetseg/configs/ecdetpose/examples/ecdetpose_m_vehicle_31kpts.yml \
  --resume /path/to/best.pth \
  --input-size 544 960 \
  --opset 18 \
  --check \
  --simplify
```

`--input-size` overrides `eval_spatial_size` before model construction. The
encoder positional embeddings and decoder anchors are therefore created for
the requested ONNX resolution. Both dimensions must be divisible by the ViT
patch size `16`; `544 × 960` also satisfies the recommended divisibility by
`32`.

Checkpoint `decoder.anchors` and `decoder.valid_mask` entries are derived from
the training resolution rather than learned parameters. The exporter therefore
does not restore those two buffers: it retains the fresh versions generated for
`--input-size`. This permits exporting a checkpoint trained at `640 × 640` as a
fixed `544 × 960` graph without weakening strict validation of learned weights.

When `--resume` is supplied, the exporter reads `num_classes` from the
checkpoint classification heads **before** constructing the model. This lets a
multi-class checkpoint be exported even if the referenced vehicle example has
`num_classes: 1`. For example, a checkpoint whose score-head weights have shape
`[5, hidden_dim]` is exported as a five-class model. You can add
`--num-classes 5` as an explicit assertion; the exporter rejects it if it does
not match the checkpoint. The export config must still match the checkpoint's
S/M architecture, decoder depth/dimensions, and keypoint-head settings.

An ECDetPose graph has five named outputs: `labels`, `boxes`, `scores`,
`keypoints`, and `keypoint_scores`. `orig_target_sizes` uses `[width, height]`,
whereas `eval_spatial_size` uses `[height, width]`; the exporter performs this
conversion explicitly so non-square exports scale bbox X/Y and keypoint X/Y
correctly.

The exported graph supports a dynamic batch dimension. Its spatial resolution
is intentionally fixed to `eval_spatial_size`, because the backbone and decoder
precompute resolution-dependent feature shapes and anchors in deploy mode. Use
a separate export for each required inference resolution. Shape-related
`TracerWarning` messages from architecture-static branches do not indicate an
export failure; `--check` validates the resulting graph. The exporter avoids
Tensor-to-Python conversions in its multiscale feature sizing, dynamic-batch
anchor expansion, and pose-head loop so these warnings are not emitted by those
paths.

During training, every decoder layer uses the pose head with the same layer
index. During evaluation and ONNX export, the decoder returns only the selected
`eval_idx` feature tensor, so pose inference explicitly applies the
`eval_idx` coordinate and visibility heads to that single tensor. This avoids
both an out-of-bounds access (one returned feature versus several configured
heads) and the incorrect use of layer-zero pose weights for the final decoder
prediction.

The exporter uses PyTorch's legacy TorchScript ONNX path by default. This is
intentional: some PyTorch releases fail in the newer Dynamo exporter with
`No ONNX function found for aten.mul.Scalar` on real-valued scalar constants.
The bbox-relative pose decoder also represents its `0.5` factor as a tensor
constant, which is portable across both exporters. Use `--dynamo` only when the
installed PyTorch/ONNX stack supports the entire graph; it is not required for
a valid pose export.

### Rectangular input resolution

All configuration and transform sizes use `[height, width]`, not
`[width, height]`. For 960-pixel-wide 16:9 input, use `[544, 960]` rather than
`[960, 540]`: `540` is not divisible by the ViT patch size (`16`), while `544`
is. The adapter now rejects non-aligned input with a direct error and suggests
the next valid dimensions instead of failing later with an opaque tensor reshape
error. You do not need to keep `640 × 640`; rectangular training is supported
when both dimensions are divisible by 16. For the cleanest feature pyramid and
deployment behavior, dimensions divisible by 32 are recommended.

The training resize must match the model resolution, for example:

```yaml
eval_spatial_size: [544, 960]

train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: ResizeWithKeypoints, size: [544, 960]}
        # remaining pose-aware transforms...

val_dataloader:
  dataset:
    transforms:
      ops:
        - {type: ResizeWithKeypoints, size: [544, 960]}
        # remaining validation transforms...
```

`ResizeWithKeypoints` applies the same horizontal and vertical resize factors to
the image, bboxes, and keypoint coordinates, so rectangular resizing preserves
their alignment. It may change image aspect ratio if the source ratio differs;
use training-GT visualizations to confirm whether that geometric distortion is
acceptable for the dataset.

The encoder and decoder cache positional embeddings and anchors for
`eval_spatial_size`. Evaluation now verifies those caches against the actual
feature-map token count and rebuilds them when the validation transform uses a
different resolution. This prevents token errors such as `510` image features
versus `400` cached positional tokens. Nevertheless, using the same resolution
for `eval_spatial_size` and the validation `ResizeWithKeypoints` is strongly
recommended because it avoids repeated rebuilding and ensures that reported
metrics correspond to the intended deployment resolution.
