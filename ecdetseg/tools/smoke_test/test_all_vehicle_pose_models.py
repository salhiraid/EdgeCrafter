#!/usr/bin/env python3
"""Load every ECDet-pose size and verify bbox inference without grading keypoints.

A trained joint checkpoint is required for a semantic detection check. Keypoint
outputs are checked only for shape and finite coordinates because an untrained
pose head is intentionally not expected to predict correct landmarks.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

ECDET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ECDET_ROOT))

from engine.core import YAMLConfig  # noqa: E402

SIZES = ('s', 'm', 'l', 'x')
EXPECTED_BACKBONES = {
    's': ('ecvitt', 192),
    'm': ('ecvittplus', 256),
    'l': ('ecvits', 384),
    'x': ('ecvitsplus', 384),
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'ema' in checkpoint:
        return checkpoint['ema']['module']
    return checkpoint.get('model', checkpoint)


def box_iou(boxes, target):
    left_top = torch.maximum(boxes[:, :2], target[:2])
    right_bottom = torch.minimum(boxes[:, 2:], target[2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(1)
    box_area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
    target_area = (target[2:] - target[:2]).clamp(min=0).prod()
    return intersection / (box_area + target_area - intersection + 1e-7)


def load_annotations(path):
    payload = json.loads(path.read_text())
    images = {item['file_name']: item['id'] for item in payload['images']}
    boxes = {}
    for annotation in payload['annotations']:
        x, y, width, height = annotation['bbox']
        boxes.setdefault(annotation['image_id'], []).append(
            [x, y, x + width, y + height])
    return images, boxes


def build_model(size, checkpoint, device):
    config_path = ECDET_ROOT / f'configs/ecdet_pose/ecdet_pose_{size}.yml'
    cfg = YAMLConfig(str(config_path))
    expected_name, expected_dim = EXPECTED_BACKBONES[size]
    backbone_cfg = cfg.yaml_cfg['ViTAdapter']
    if backbone_cfg['name'] != expected_name or backbone_cfg['embed_dim'] != expected_dim:
        raise AssertionError(
            f'{size}: expected backbone {expected_name}/{expected_dim}, got '
            f"{backbone_cfg['name']}/{backbone_cfg['embed_dim']}")
    # A full checkpoint already contains its backbone; avoid a second download.
    cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
    model = cfg.model
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model = model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    return model, postprocessor, cfg.yaml_cfg['eval_spatial_size'], expected_name


@torch.inference_mode()
def predict(model, postprocessor, image, input_size, device):
    transform = T.Compose([T.Resize(input_size), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])])
    tensor = transform(image).unsqueeze(0).to(device)
    target_size = torch.tensor([[image.width, image.height]], device=device)
    result = postprocessor(model(tensor), target_size)[0]
    if result['boxes'].ndim != 2 or result['boxes'].shape[1] != 4:
        raise AssertionError(f"invalid box shape {tuple(result['boxes'].shape)}")
    if result['keypoints'].shape[-1] != 31 * 3:
        raise AssertionError(f"invalid keypoint shape {tuple(result['keypoints'].shape)}")
    for name in ('scores', 'boxes', 'keypoints'):
        if not torch.isfinite(result[name]).all():
            raise AssertionError(f'non-finite {name} prediction')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True,
                        help='COCO JSON used only to verify bbox predictions')
    parser.add_argument('--checkpoint-dir', type=Path, required=True,
                        help='Contains ecdet_pose_{s,m,l,x}.pth trained checkpoints')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--score-threshold', type=float, default=0.25)
    parser.add_argument('--min-box-iou', type=float, default=0.3)
    parser.add_argument('--report', type=Path, default=Path('outputs/ecdet_pose_all_sizes_test.json'))
    args = parser.parse_args()

    device = torch.device(args.device)
    image_ids, ground_truth = load_annotations(args.annotations)
    image_paths = sorted(path for path in args.images.iterdir()
                         if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError(f'No supported images in {args.images}')

    report = {'note': 'Keypoint accuracy is intentionally not asserted.', 'models': {}}
    for size in SIZES:
        checkpoint = args.checkpoint_dir / f'ecdet_pose_{size}.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing {size.upper()} checkpoint: {checkpoint}')
        model, postprocessor, input_size, backbone = build_model(size, checkpoint, device)
        model_report = {'backbone': backbone, 'images': []}
        for image_path in image_paths:
            with Image.open(image_path) as source:
                image = source.convert('RGB')
            prediction = predict(model, postprocessor, image, input_size, device)
            selected = prediction['scores'] >= args.score_threshold
            predicted_boxes = prediction['boxes'][selected].cpu()
            image_id = image_ids.get(image_path.name)
            expected_boxes = ground_truth.get(image_id, [])
            best_ious = []
            for expected in expected_boxes:
                if not len(predicted_boxes):
                    best_ious.append(0.0)
                else:
                    best_ious.append(float(box_iou(
                        predicted_boxes, torch.tensor(expected)).max()))
            if expected_boxes and min(best_ious) < args.min_box_iou:
                raise AssertionError(
                    f'{size}/{image_path.name}: bbox IoU {best_ious} is below '
                    f'{args.min_box_iou}; use a trained joint checkpoint')
            model_report['images'].append({
                'file': image_path.name,
                'detections': int(selected.sum()),
                'best_bbox_iou': best_ious,
                'keypoint_shape': list(prediction['keypoints'].shape),
                'keypoint_accuracy_checked': False,
            })
        report['models'][size] = model_report
        print(f'PASS {size.upper()}: {backbone}, {len(image_paths)} image(s)')
        del model, postprocessor
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {args.report}')


if __name__ == '__main__':
    main()
