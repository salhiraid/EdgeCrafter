# ECDetPose Model Zoo

| Model | Source Detection Baseline | Keypoints | Init Checkpoint | SHA-256 | Detection Equivalence | Keypoint State |
|---|---|---:|---|---|---|---|
| ECDetPose-S | ECDet-S | 31 | `checkpoints/ecdetpose_s_init_from_ecdet.pth` | Not generated | Not run | Detection pretrained; vehicle-keypoint head randomly initialized |
| ECDetPose-M | ECDet-M | 31 | `checkpoints/ecdetpose_m_init_from_ecdet.pth` | Not generated | Not run | Detection pretrained; vehicle-keypoint head randomly initialized |
| ECDetPose-L | ECDet-L | 31 | `checkpoints/ecdetpose_l_init_from_ecdet.pth` | Not generated | Not run | Detection pretrained; vehicle-keypoint head randomly initialized |
| ECDetPose-X | ECDet-X | 31 | `checkpoints/ecdetpose_x_init_from_ecdet.pth` | Not generated | Not run | Detection pretrained; vehicle-keypoint head randomly initialized |

Use `ecdetseg/tools/model_info.py` to compute measured parameter counts, checkpoint sizes, and FLOPs/MACs in an environment with PyTorch and optional `calflops` installed:

```bash
python ecdetseg/tools/model_info.py \
  --configs \
  ecdetseg/configs/ecdetpose/ecdetpose_s.yml \
  ecdetseg/configs/ecdetpose/ecdetpose_m.yml \
  ecdetseg/configs/ecdetpose/ecdetpose_l.yml \
  ecdetseg/configs/ecdetpose/ecdetpose_x.yml
```
