import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(dataset, num_keypoints=None):
    images = {img["id"]: img for img in dataset.get("images", [])}
    categories = {cat["id"]: cat for cat in dataset.get("categories", [])}
    inferred = [
        len(cat.get("keypoints", []))
        for cat in categories.values()
        if cat.get("keypoints")
    ]
    if num_keypoints is None:
        num_keypoints = inferred[0] if inferred else None

    report = {
        "number_of_images": len(images),
        "number_of_annotations": len(dataset.get("annotations", [])),
        "number_of_categories": len(categories),
        "number_of_keypoints": num_keypoints,
        "objects_with_missing_keypoints": 0,
        "malformed_keypoint_arrays": [],
        "visibility_distribution": Counter(),
        "visible_keypoints_outside_boxes": 0,
        "keypoints_outside_image_boundaries": 0,
        "empty_images": 0,
        "crowd_annotations": 0,
        "per_category_counts": Counter(),
    }

    anns_by_image = defaultdict(list)
    for ann in dataset.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)
        report["per_category_counts"][str(ann.get("category_id"))] += 1
        if ann.get("iscrowd", 0):
            report["crowd_annotations"] += 1

        keypoints = ann.get("keypoints")
        if not keypoints:
            report["objects_with_missing_keypoints"] += 1
            continue
        if num_keypoints is None or len(keypoints) != num_keypoints * 3:
            report["malformed_keypoint_arrays"].append({
                "image_id": ann.get("image_id"),
                "annotation_id": ann.get("id"),
                "length": len(keypoints),
                "expected": None if num_keypoints is None else num_keypoints * 3,
            })
            continue

        img = images.get(ann["image_id"], {})
        img_w, img_h = img.get("width", 0), img.get("height", 0)
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        for i in range(num_keypoints):
            px, py, v = keypoints[3 * i: 3 * i + 3]
            report["visibility_distribution"][str(int(v))] += 1
            if v <= 0:
                continue
            if not (0 <= px <= img_w and 0 <= py <= img_h):
                report["keypoints_outside_image_boundaries"] += 1
            if v == 2 and not (x <= px <= x + w and y <= py <= y + h):
                report["visible_keypoints_outside_boxes"] += 1

    for image_id in images:
        if not anns_by_image.get(image_id):
            report["empty_images"] += 1

    report["visibility_distribution"] = dict(report["visibility_distribution"])
    report["per_category_counts"] = dict(report["per_category_counts"])
    return report


def export_visualizations(dataset, image_root, output_dir, num_keypoints, limit):
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for --visualize-output. Install pillow or omit visualization arguments.")
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = {img["id"]: img for img in dataset.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in dataset.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    saved = 0
    for image_id, anns in anns_by_image.items():
        if saved >= limit:
            break
        info = images.get(image_id)
        if not info:
            continue
        path = image_root / info["file_name"]
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for ann in anns:
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)
            keypoints = ann.get("keypoints") or []
            if len(keypoints) != num_keypoints * 3:
                continue
            for i in range(num_keypoints):
                px, py, v = keypoints[3 * i: 3 * i + 3]
                if v > 0:
                    color = "red" if v == 2 else "orange"
                    draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)
        image.save(output_dir / f"{image_id}.jpg")
        saved += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--num-keypoints", type=int, default=None)
    parser.add_argument("--image-root")
    parser.add_argument("--visualize-output")
    parser.add_argument("--visualize-limit", type=int, default=16)
    args = parser.parse_args()

    dataset = load_json(args.ann_file)
    report = validate(dataset, args.num_keypoints)
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.image_root and args.visualize_output:
        export_visualizations(
            dataset,
            args.image_root,
            args.visualize_output,
            report["number_of_keypoints"],
            args.visualize_limit,
        )


if __name__ == "__main__":
    main()
