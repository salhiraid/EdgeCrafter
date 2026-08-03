<h3 align="center">
  <b>English</b> | <a href="README_zh.md">简体中文</a>
</h3>

## 📋 Table of Contents
- [Model Zoo](#-model-zoo)
- [Dataset Preparation](#-dataset-preparation)
- [Model Configuration](#-model-configuration)
- [Usage](#-usage)
- [Tools](#-tools)

---

## 🏆 Model Zoo

### COCO2017 Validation Results (Keypoints)


| Model | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----:|:--:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECPose-S** | 640 | 68.9 |  10 | 30 | 5.54 | [config](configs/ecpose/ecpose_s_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth) |
| **ECPose-M** | 640 | 72.4 |  20 | 63 | 9.25 | [config](configs/ecpose/ecpose_m_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_m.pth) |
| **ECPose-L** | 640 | 73.5 |  34 | 112 | 11.83 | [config](configs/ecpose/ecpose_l_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_l.pth) |
| **ECPose-X** | 640 | 74.8 |  51 | 172 | 14.31 | [config](configs/ecpose/ecpose_x_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_x.pth) |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

### ⚡ Quick Start (Inference)
The easiest way to test ECPose is to run inference on a sample image using a pre-trained model.
```bash
# 1. Download a pre-trained model (e.g., ECPose-L)
wget https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_l.pth
# 2. Run PyTorch inference
# Make sure to replace `path/to/your/image.jpg` with an actual image path
python tools/inference/torch_inf.py -c configs/ecpose/ecpose_l_coco.yml -r ecpose_l.pth -i path/to/your/image.jpg
```

## 📁 Dataset Preparation

### COCO2017 Keypoints

1. Download COCO2017 and keypoint annotations from the official COCO website.
2. Organize data as:

```text
/path/to/COCO2017/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
└── val2017/
```

3. Update paths in [`configs/dataset/coco_pose.yml`](./configs/dataset/coco_pose.yml):

```yaml
train_dataloader:
  dataset:
    img_folder: /path/to/COCO2017/train2017
    ann_file: /path/to/COCO2017/annotations/person_keypoints_train2017.json

val_dataloader:
  dataset:
    img_folder: /path/to/COCO2017/val2017
    ann_file: /path/to/COCO2017/annotations/person_keypoints_val2017.json
```

### Custom Dataset (COCO Keypoints Format)

Use the same format as COCO keypoints and adapt `configs/dataset/coco_pose.yml`:

- set `img_folder` / `ann_file` to your dataset paths
- keep `task: pose`
- adjust `num_classes` and remapping behavior if needed

### Vehicle Bounding Boxes + 31 Keypoints

The joint ECDet/ECPose variants are available in all four sizes:
`configs/ecvehicle/ecvehicle_{s,m,l,x}.yml`. They share the EC backbone and
decoder and train three heads end-to-end: vehicle classification (VFL), an
independent bounding-box head (L1 + GIoU), and the 31-point pose head (visible
point L1 + OKS). Update the image/annotation paths and taxonomy in
[`configs/dataset/vehicle_keypoints.yml`](./configs/dataset/vehicle_keypoints.yml),
then set `num_classes` in both that file and `DETRPoseCriterion`. Category IDs
must be contiguous and zero-based (`car=0`, `bus=1`, etc.). Every annotation
contains 93 values in COCO order, `[x0,y0,v0,...,x30,y30,v30]`, where visibility
is 0 (not labelled), 1 (labelled/occluded), or 2 (labelled/visible).

All classes can use the full layout. To configure a subset for a vehicle class,
set `category_keypoint_indices` on both train and validation datasets. Missing
points are masked to `(0,0,0)` and do not contribute to keypoint matching or
losses. For example: `{0: [0,1,...,30], 1: [0,1,2,3,26,27]}`. The canonical
order is:

```text
 0 front_window_edge_right       1 front_window_edge_left
 2 front_light_left              3 front_light_right
 4 front_windshield_up_left      5 front_windshield_up_right
 6 front_central_up_left         7 front_central_up_right
 8 front_low_left                9 front_low_right
10 front_plate_right            11 front_plate_left
12 rear_light_left              13 rear_light_right
14 rear_windshield_up_left      15 rear_windshield_up_right
16 rear_plate_right             17 rear_plate_left
18 rear_low_left                19 rear_low_right
20 front_windshield_low_right   21 front_windshield_low_left
22 front_up_right               23 front_up_left
24 rear_up_right                25 rear_up_left
26 front_wheel_left             27 front_wheel_right
28 rear_wheel_left              29 rear_wheel_right
30 rear_seat_end
```

Unlike deriving a box from the minimum and maximum keypoint coordinates, this
configuration uses an independent learnable bbox branch on each decoder layer.
The branch is optimized with L1 and GIoU losses, while the keypoint branch is
optimized with keypoint and OKS losses.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecvehicle/ecvehicle_s.yml --use-amp --seed=0

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecvehicle/ecvehicle_s.yml --test-only -r /path/to/model.pth
```

#### Three-image smoke test

Run the complete generate → train → evaluate → log workflow on three images
with deterministic pseudo vehicle annotations:

```bash
cd ecpose
python tools/smoke_test/run_vehicle_smoke_test.py --device cpu
```

To use your own three images and one COCO JSON instead, run:

```bash
python tools/smoke_test/run_vehicle_smoke_test.py --device cpu \
  --images /path/to/three/images --annotations /path/to/annotations.json
```

The smoke test uses a deliberately small two-block backbone, 128×128 inputs,
five object queries, 31 keypoints, and one epoch. It validates the pipeline; its synthetic
annotations and metrics are **not** an accuracy benchmark. Use `--device cuda:0`
for a GPU run, or `--skip-download` to reuse `smoke_data/`. If the COCO image
host is unavailable, the downloader creates three deterministic local fallback
images; pass `--strict-download` directly to the downloader to disable fallback.

Artifacts are written under `outputs/ecvehicle_smoke/`:

- `log.txt`: one JSON object per epoch, including every averaged loss
- `training.log`: timestamped text/JSON training records
- `tensorboard/`: scalar events for total/component losses, learning rates, and
  epoch summaries
- `checkpoint.pth`: resumable smoke-test checkpoint

Launch TensorBoard with:

```bash
tensorboard --logdir outputs/ecvehicle_smoke/tensorboard
```

---

## 🔌 Model Configuration

Model configs are in [`configs/ecpose`](./configs/ecpose/), e.g. [ecpose_s_coco.yml](./configs/ecpose/ecpose_s_coco.yml):

```yaml
__include__: [
  '../dataset/coco_pose.yml',
  'ecpose.yml'
]


output_dir: outputs/ecpose_s

ViTAdapter:
  name: ecpose_vitt
  embed_dim: 192
  weights_path: ecvits/ecpose_vitt.pth    # Pretrained backbone; automatically downloaded on first use.
  interaction_indexes: [10, 11]
  num_heads: 3

eval_spatial_size: [640, 640]   # Input Resolution

epoches: 92  # Total training epochs.
grad_accum_steps: 1


## Optimizer
optimizer:
  type: AdamW
  params: 
    -
      params: '^(?=.*.backbone)(?!.*(?:norm|bn|bias)).*$'  # Backbone parameters excluding normalization layers and bias
      lr: 0.000025
    -
      params: '^(?=.*.backbone)(?=.*(?:norm|bn|bias)).*$'  # Backbone normalization layers (norm/bn) and bias parameters
      lr: 0.000025
      weight_decay: 0.
    - 
      params: '^(?!.*\.backbone)(?=.*(?:norm|bn|bias)).*$'  # Non-backbone normalization layers and bias parameters
      weight_decay: 0.

  lr: 0.0005  # Base learning rate for non-backbone parameters
  betas: [0.9, 0.999]
  weight_decay: 0.0001

train_dataloader: 
  dataset: 
    transforms:
      ops:
        - {type: PoseMosaic, output_size: 320}
        - {type: MixUpCopyPaste, mixup_prob: 0.5}
        - {type: RandomZoomOut, p: 0.5}
        - {type: RandomHorizontalFlip}
        - {type: ColorJitter}
        - {type: Resize, size: [640, 640]}
        - {type: ToTensor}
        - {type: Normalize, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]} 
      mosaic_prob: 0.5  # Probability of applying Mosaic augmentation
      policy:
        epoch: [4, 45, 90]   # Mosaic starts at epoch 4, stops at epoch 45, and all augmentations are disabled at epoch 90.
       
  collate_fn:
    stop_epoch: 90  # all augmentations are disabled at epoch 90.
```

---

## 🎮 Usage

### Training

```bash
# Generic (single node, 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --use-amp --seed=0
```

Replace `{SIZE}` with `s`, `m`, `l`, or `x`.

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --test-only -r /path/to/model.pth
```

### Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --use-amp --seed=0 -t /path/to/model.pth
```

---

## 🔧 Tools

### PyTorch Inference (image/video)

```bash
python tools/inference/torch_inf.py \
  -c configs/ecpose/ecpose_{SIZE}_coco.yml \
  -r ecpose_{SIZE}.pth \
  -i /path/to/image_or_video
```

Optional flags: `-d cuda:0`, `-t 0.4`, `--no-skeleton`.

### ONNX Export

```bash
python tools/deployment/export_onnx.py \
  -c configs/ecpose/ecpose_{SIZE}_coco.yml \
  -r ecpose_{SIZE}.pth \
  --check --simplify
```

### ONNX Inference (image/video)

```bash
python tools/inference/onnx_inf.py \
  --onnx ecpose_{SIZE}.onnx \
  --input /path/to/image_or_video \
  --device cuda
```
