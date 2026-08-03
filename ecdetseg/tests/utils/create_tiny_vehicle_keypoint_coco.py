import json
from pathlib import Path

from PIL import Image, ImageDraw


def create_tiny_vehicle_keypoint_coco(root, num_images=4, num_keypoints=31):
    root = Path(root)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    ann_id = 1
    for i in range(num_images):
        width, height = 96, 64
        file_name = f"vehicle_{i:03d}.png"
        img = Image.new("RGB", (width, height), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        images.append({"id": i + 1, "file_name": file_name, "width": width, "height": height})
        if i == num_images - 1:
            img.save(image_dir / file_name)
            continue
        for obj_idx in range(1 + (i % 2)):
            x, y, w, h = 8 + obj_idx * 36, 10 + i * 2, 30, 24
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0))
            keypoints = []
            for k in range(num_keypoints):
                px = x + 2 + (k % 6) * (w - 4) / 5
                py = y + 2 + (k // 6) * (h - 4) / 5
                v = 2
                if k % 11 == 0:
                    v = 1
                if k % 13 == 0:
                    v = 0
                    px, py = 0, 0
                keypoints.extend([round(px, 3), round(py, 3), v])
            annotations.append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
            })
            ann_id += 1
        img.save(image_dir / file_name)

    data = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "vehicle",
            "keypoints": [f"kp_{i + 1:02d}" for i in range(num_keypoints)],
            "skeleton": [],
        }],
    }
    ann_file = root / "annotations.json"
    ann_file.write_text(json.dumps(data), encoding="utf-8")
    return image_dir, ann_file


if __name__ == "__main__":
    create_tiny_vehicle_keypoint_coco("tiny_vehicle_keypoints")
