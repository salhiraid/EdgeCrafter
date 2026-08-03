#!/usr/bin/env python3
"""Download five public COCO images and create tiny COCO keypoint annotations.

The annotations are deterministic pseudo-labels (a centered vehicle-like box with
four corner landmarks). They exercise the complete training/evaluation pipeline;
they are not intended to measure model accuracy.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_IDS = [397133, 37777, 252219, 87038, 174482]
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
    inset_x, inset_y = box_w * 0.12, box_h * 0.18
    keypoints = [
        x + inset_x, y + inset_y, 2,
        x + box_w - inset_x, y + inset_y, 2,
        x + box_w - inset_x, y + box_h - inset_y, 2,
        x + inset_x, y + box_h - inset_y, 2,
    ]
    return {
        "id": image_id,
        "image_id": image_id,
        "category_id": 0,
        "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
        "area": round(box_w * box_h, 2),
        "iscrowd": 0,
        "num_keypoints": 4,
        "keypoints": [round(value, 2) if index % 3 != 2 else int(value)
                      for index, value in enumerate(keypoints)],
    }


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
        "info": {"description": "EdgeCrafter five-image smoke-test dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 0,
            "name": "vehicle",
            "supercategory": "vehicle",
            "keypoints": ["front_left", "front_right", "rear_right", "rear_left"],
            "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]],
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
