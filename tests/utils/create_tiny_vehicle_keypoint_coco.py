import json
from pathlib import Path

from PIL import Image, ImageDraw


def create_tiny_vehicle_keypoint_coco(root, num_keypoints=31):
    root = Path(root)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    images = []

    for image_id in range(4):
        image = Image.new("RGB", (128, 96), color=(30, 30, 30))
        draw = ImageDraw.Draw(image)
        file_name = f"{image_id:04d}.jpg"
        images.append({"id": image_id, "file_name": file_name, "width": 128, "height": 96})
        if image_id != 2:
            for obj_idx in range(1 + (image_id == 1)):
                x = 10 + obj_idx * 45
                y = 12 + image_id * 8
                w = 42
                h = 30
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0))
                keypoints = []
                for k in range(num_keypoints):
                    px = x + 2 + (k % 8) * 5
                    py = y + 2 + (k // 8) * 6
                    v = 2
                    if k % 11 == 0:
                        v = 1
                    if k % 13 == 0:
                        v = 0
                    keypoints.extend([px, py, v])
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": num_keypoints,
                })
        image.save(image_dir / file_name)

    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "vehicle",
            "keypoints": [f"kp_{i + 1}" for i in range(num_keypoints)],
            "skeleton": [],
        }],
    }
    ann_file = root / "annotations.json"
    ann_file.write_text(json.dumps(dataset), encoding="utf-8")
    return image_dir, ann_file


if __name__ == "__main__":
    create_tiny_vehicle_keypoint_coco(Path("tiny_vehicle_keypoints"))
