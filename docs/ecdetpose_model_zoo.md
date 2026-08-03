# ECDetPose Model Zoo

| Model | Source Detection Baseline | Keypoints | Init Checkpoint | SHA-256 | Detection Equivalence | Keypoint State |
|---|---|---:|---|---|---|---|
| ECDetPose-S smoke | ECDet-S random-source smoke | 31 | `checkpoints/ecdetpose_s_cli_init_from_random_ecdet.pth` | `C28C035AC8441D03CD4D68CA4F43DE6A8A1499BE1DB8400D43FD21169BC18BF5` | Passed: logits `0.0`, boxes `0.0` | Detection weights copied; vehicle-keypoint head randomly initialized |
| ECDetPose-M | ECDet-M | 31 | `checkpoints/ecdetpose_m_init_from_ecdet.pth` | Not generated | Not run | Detection pretrained; vehicle-keypoint head randomly initialized |
| ECDetPose-L | ECDet-L | 31 | `checkpoints/ecdetpose_l_init_from_ecdet.pth` | Not generated | Not run | Detection pretrained; vehicle-keypoint head randomly initialized |
| ECDetPose-X | ECDet-X | 31 | `checkpoints/ecdetpose_x_init_from_ecdet.pth` | Not generated | Not run | Detection pretrained; vehicle-keypoint head randomly initialized |

Measured ECDetPose-S smoke model info:

| Model | Params | MACs | FLOPs | Input | Queries | Decoder Layers |
|---|---:|---:|---:|---|---:|---:|
| ECDetPose-S | 10.25M | 13.05 GMACs | 26.21 GFLOPs | 640x640 | 300 | 4 |

The S smoke checkpoint validates conversion mechanics and output format. It is not an official pretrained detection checkpoint because the official ECDet `.pth` files were not present locally.

Use `ecdetseg/tools/model_info.py` to compute measured parameter counts, checkpoint sizes, and FLOPs/MACs in an environment with PyTorch and optional `calflops` installed:

```bash
python ecdetseg/tools/model_info.py \
  --configs \
  ecdetseg/configs/ecdetpose/ecdetpose_s.yml \
  ecdetseg/configs/ecdetpose/ecdetpose_m.yml \
  ecdetseg/configs/ecdetpose/ecdetpose_l.yml \
  ecdetseg/configs/ecdetpose/ecdetpose_x.yml
```
