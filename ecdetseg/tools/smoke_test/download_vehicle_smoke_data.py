#!/usr/bin/env python3
"""Download three public COCO images and create tiny COCO keypoint annotations.

The annotations are deterministic pseudo-labels (a centered vehicle-like box with
31 vehicle landmarks). They exercise the complete training/evaluation pipeline;
they are not intended to measure model accuracy.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_IDS = [397133, 37777, 252219]
BASE_URL = "https://images.cocodataset.org/val2017/{filename}"


def download(url: str, destination: Path, retries: int = 3) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "EdgeCrafter-smoke-test/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            return
        except (OSError, urllib.error.URLError) as error:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {error}") from error
            time.sleep(attempt)


def make_annotation(image_id: int, width: int, height: int) -> dict:
    box_w, box_h = width * 0.6, height * 0.45
    x, y = (width - box_w) / 2, (height - box_h) / 2
    keypoints = []
    for index in range(31):
        column, row = index % 8, index // 8
        keypoints.extend([x + box_w * (0.08 + column * 0.12),
                          y + box_h * (0.12 + row * 0.24), 2])
    return {"id": image_id, "image_id": image_id, "category_id": 0,
            "bbox": [x, y, box_w, box_h], "area": box_w * box_h,
            "iscrowd": 0, "num_keypoints": 31, "keypoints": keypoints}


def create_fallback_image(destination: Path, image_id: int) -> None:
    """Create a deterministic local image when the public host is unavailable."""
    width, height = 320, 240
    image = Image.new('RGB', (width, height), color=(35, 55, 75))
    draw = ImageDraw.Draw(image)
    color = (80 + image_id % 150, 90, 180)
    draw.rectangle((64, 78, 256, 186), fill=color, outline=(245, 245, 245), width=3)
    draw.ellipse((82, 168, 122, 208), fill=(20, 20, 20))
    draw.ellipse((198, 168, 238, 208), fill=(20, 20, 20))
    image.save(destination, quality=90)


def build_dataset(root: Path, strict_download: bool = False) -> None:
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    for image_id in IMAGE_IDS:
        filename = f"{image_id:012d}.jpg"
        image_path = images_dir / filename
        try:
            download(BASE_URL.format(filename=filename), image_path)
        except RuntimeError as error:
            if strict_download:
                raise
            print(f"warning: {error}")
            print(f"creating deterministic fallback image: {image_path}")
            create_fallback_image(image_path, image_id)
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": filename, "width": width, "height": height})
        annotations.append(make_annotation(image_id, width, height))
        print(f"ready: {image_path} ({width}x{height})")

    dataset = {
        "info": {"description": "EdgeCrafter three-image 31-keypoint smoke-test dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 0,
            "name": "vehicle",
            "supercategory": "vehicle",
            "keypoints": ['front_window_edge_right', 'front_window_edge_left', 'front_light_left', 'front_light_right', 'front_windshield_up_left', 'front_windshield_up_right', 'front_central_up_left', 'front_central_up_right', 'front_low_left', 'front_low_right', 'front_plate_right', 'front_plate_left', 'rear_light_left', 'rear_light_right', 'rear_windshield_up_left', 'rear_windshield_up_right', 'rear_plate_right', 'rear_plate_left', 'rear_low_left', 'rear_low_right', 'front_windshield_low_right', 'front_windshield_low_left', 'front_up_right', 'front_up_left', 'rear_up_right', 'rear_up_left', 'front_wheel_left', 'front_wheel_right', 'rear_wheel_left', 'rear_wheel_right', 'rear_seat_end'],
            "skeleton": [],
        }],
    }
    for split in ("train", "val"):
        destination = annotations_dir / f"{split}.json"
        destination.write_text(json.dumps(dataset, indent=2) + "\n")
        print(f"wrote: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("smoke_data"))
    parser.add_argument("--strict-download", action="store_true",
                        help="Fail instead of generating fallback images when COCO is unreachable")
    args = parser.parse_args()
    build_dataset(args.output, strict_download=args.strict_download)
