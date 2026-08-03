#!/usr/bin/env python3
"""Create and preview S/M/L/X pose models from downloadable ECDet weights.

Detection is produced by the released ECDet checkpoint. A new ECDet-pose model
receives every compatible detector weight (backbone and encoder) while its pose
decoder remains randomly initialized, so its keypoints are expected to be wrong
until training. Both states are saved together as an initialization bundle.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.core import YAMLConfig  # noqa: E402

SIZES = ('s', 'm', 'l', 'x')
URL = 'https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_{size}.pth'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
COLORS = [(255, 0, 0), (0, 180, 0), (0, 80, 255), (255, 170, 0)]


def download(url, destination):
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url} -> {destination}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'EdgeCrafter-pose-init/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def state_from_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint.get('model', checkpoint)


def collect_images(inputs):
    paths = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(item for item in path.iterdir()
                                if item.suffix.lower() in IMAGE_SUFFIXES))
        elif path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f'Not an image or image directory: {path}')
    if not paths:
        raise ValueError('No images found')
    return list(dict.fromkeys(path.resolve() for path in paths))


def build_models(size, detector_checkpoint, device, output_dir):
    detector_cfg = YAMLConfig(str(ROOT / f'configs/ecdet/ecdet_{size}.yml'))
    detector_cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
    detector = detector_cfg.model
    detector_state = state_from_checkpoint(detector_checkpoint)
    detector.load_state_dict(detector_state, strict=True)

    pose_cfg = YAMLConfig(str(ROOT / f'configs/ecdet_pose/ecdet_pose_{size}.yml'))
    pose_cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
    pose = pose_cfg.model
    pose_state = pose.state_dict()
    compatible = {
        name: value for name, value in detector_state.items()
        if name in pose_state and pose_state[name].shape == value.shape
    }
    incompatible = pose.load_state_dict(compatible, strict=False)
    required_prefixes = ('backbone.', 'encoder.')
    missing_shared = [name for name in pose_state
                      if name.startswith(required_prefixes) and name not in compatible]
    if missing_shared:
        raise RuntimeError(f'{size}: detector did not initialize shared weights: {missing_shared[:5]}')

    bundle_path = output_dir / f'ecdet_pose_{size}_initialized.pth'
    torch.save({
        'model': pose.state_dict(),
        'detector_model': detector_state,
        'source_detector': str(detector_checkpoint),
        'initialized_keys': sorted(compatible),
        'random_pose_keys': sorted(incompatible.missing_keys),
        'note': 'Pose decoder/keypoint predictions require training.',
    }, bundle_path)
    return (detector.to(device).eval(), detector_cfg.postprocessor.to(device).eval(),
            pose.to(device).eval(), pose_cfg.postprocessor.to(device).eval(),
            detector_cfg.yaml_cfg['eval_spatial_size'], bundle_path,
            len(compatible), len(incompatible.missing_keys))


@torch.inference_mode()
def infer(model, postprocessor, image, input_size, device):
    transform = T.Compose([T.Resize(input_size), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])])
    sample = transform(image).unsqueeze(0).to(device)
    size = torch.tensor([[image.width, image.height]], device=device)
    return postprocessor(model(sample), size)[0]


def closest_pose(det_box, pose_boxes):
    det_center = (det_box[:2] + det_box[2:]) / 2
    pose_centers = (pose_boxes[:, :2] + pose_boxes[:, 2:]) / 2
    return int(((pose_centers - det_center) ** 2).sum(1).argmin())


def draw_preview(image, detections, poses, threshold):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    keep = torch.nonzero(detections['scores'] >= threshold).flatten()
    for number, index in enumerate(keep.tolist()):
        box = detections['boxes'][index].cpu()
        color = COLORS[number % len(COLORS)]
        draw.rectangle(box.tolist(), outline=color, width=3)
        draw.text((float(box[0]), float(box[1])),
                  f"det {float(detections['scores'][index]):.2f}", fill=color)
        # Pose queries are not trained/aligned yet. Nearest-center association is
        # visualization only and must not be interpreted as a correct landmark.
        pose_index = closest_pose(box.to(poses['boxes'].device), poses['boxes'])
        keypoints = poses['keypoints'][pose_index].reshape(31, 3).cpu()
        for x, y, visibility in keypoints:
            if visibility > 0:
                radius = 2
                draw.ellipse((float(x)-radius, float(y)-radius,
                              float(x)+radius, float(y)+radius), fill=color)
    return preview, len(keep)


def main():
    parser = argparse.ArgumentParser(
        description='Download ECDet S/M/L/X, create pose initializations, and preview images.')
    parser.add_argument('images', nargs='+', help='Copy/paste image paths and/or folders')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/ecdet_pose_initialization'))
    parser.add_argument('--score-threshold', type=float, default=0.4)
    parser.add_argument('--sizes', nargs='+', choices=SIZES, default=list(SIZES))
    args = parser.parse_args()

    images = collect_images(args.images)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'warning': 'Detection is pretrained; pose predictions are untrained.', 'models': {}}
    for size in args.sizes:
        checkpoint = args.checkpoint_dir / f'ecdet_{size}.pth'
        download(URL.format(size=size), checkpoint)
        (detector, detector_post, pose, pose_post, input_size, bundle,
         initialized, random_keys) = build_models(size, checkpoint, device, args.output_dir)
        entries = []
        for path in images:
            with Image.open(path) as source:
                image = source.convert('RGB')
            detections = infer(detector, detector_post, image, input_size, device)
            poses = infer(pose, pose_post, image, input_size, device)
            preview, count = draw_preview(image, detections, poses, args.score_threshold)
            destination = args.output_dir / f'{path.stem}_ecdet_pose_{size}.jpg'
            preview.save(destination, quality=95)
            entries.append({'image': str(path), 'preview': str(destination),
                            'pretrained_detections': count,
                            'keypoint_accuracy_expected': False})
            print(f'{size.upper()}: {path.name}: {count} pretrained detections -> {destination}')
        report['models'][size] = {'bundle': str(bundle), 'initialized_keys': initialized,
                                  'random_pose_keys': random_keys, 'images': entries}
        del detector, detector_post, pose, pose_post
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report_path = args.output_dir / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {report_path}')


if __name__ == '__main__':
    main()
