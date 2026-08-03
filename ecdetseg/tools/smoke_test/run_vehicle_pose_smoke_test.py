#!/usr/bin/env python3
"""Run one ECDet+pose training epoch on exactly three COCO-format images."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/ecdet_pose_smoke'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data = root / 'smoke_data'
    if bool(args.images) != bool(args.annotations):
        parser.error('--images and --annotations must be provided together')
    if args.images is None:
        subprocess.run([sys.executable, str(Path(__file__).with_name('download_vehicle_smoke_data.py')),
                        '--output', str(data)], check=True)
    images = args.images.resolve() if args.images else data / 'images'
    annotations = args.annotations.resolve() if args.annotations else data / 'annotations/train.json'
    payload = json.loads(annotations.read_text())
    if len(payload.get('images', [])) != 3:
        raise ValueError('The smoke test requires exactly three images')
    for annotation in payload.get('annotations', []):
        if len(annotation.get('keypoints', [])) != 93:
            raise ValueError(f"Annotation {annotation.get('id')} does not contain 31 x,y,v triples")
    command = [sys.executable, 'train.py', '-c', 'configs/ecdet_pose/ecdet_pose_smoke.yml',
               '--device', args.device, '--output-dir', str(args.output_dir.resolve()), '-u',
               f'train_dataloader.dataset.img_folder={images}',
               f'train_dataloader.dataset.ann_file={annotations}',
               f'val_dataloader.dataset.img_folder={images}',
               f'val_dataloader.dataset.ann_file={annotations}']
    env = os.environ.copy(); env.setdefault('OMP_NUM_THREADS', '1')
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
