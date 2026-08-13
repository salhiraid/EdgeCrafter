"""Run ECDetPose on a COCO dataset and export predictions/visualizations.

The COCO JSON is used as the source of image ids and file names. Prediction
JSON coordinates are always written in the original-image coordinate system.
Visualizations can instead be rendered at either the original resolution or
the model inference resolution.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TVF

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import engine  # noqa: E402,F401
from engine.core import YAMLConfig  # noqa: E402


def parse_size(value):
    """Parse HEIGHTxWIDTH, matching EdgeCrafter's eval_spatial_size order."""
    if value is None:
        return None
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2 or any(not part.strip().isdigit() for part in parts):
        raise argparse.ArgumentTypeError("size must be HEIGHTxWIDTH, for example 640x960")
    height, width = (int(part) for part in parts)
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return height, width


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    incompatible = model.load_state_dict(state, strict=False)
    missing_pose = [key for key in incompatible.missing_keys if "keypoint" in key]
    if missing_pose:
        raise RuntimeError(
            "Checkpoint does not contain the trained keypoint head. Missing keys: "
            + ", ".join(missing_pose[:10]))
    if incompatible.unexpected_keys:
        print(f"Warning: ignored {len(incompatible.unexpected_keys)} unexpected checkpoint keys")


def preprocess(image, inference_size):
    height, width = inference_size
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    tensor = TVF.to_tensor(resized)
    return TVF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), resized


def category_mapping(coco, explicit_ids):
    categories = coco.get("categories", [])
    available_ids = [int(category["id"]) for category in categories]
    if explicit_ids:
        ids = [int(value) for value in explicit_ids.split(",")]
        unknown = sorted(set(ids) - set(available_ids))
        if unknown:
            raise ValueError(f"--category-ids contains ids absent from annotations: {unknown}")
        return ids
    if not available_ids:
        raise ValueError("COCO annotation file has no categories; pass --category-ids explicitly")
    return available_ids


def selected_predictions(result, score_threshold):
    keep = result["scores"] >= score_threshold
    return {
        key: value[keep] if torch.is_tensor(value) and value.shape[:1] == keep.shape else value
        for key, value in result.items()
    }


def coco_records(result, image_id, label_to_category, keypoint_threshold):
    detections = []
    bbox_records = []
    keypoint_records = []
    keypoints = result.get("keypoints")
    keypoint_scores = result.get("keypoint_scores")

    for index, (score, label, box) in enumerate(zip(
            result["scores"], result["labels"], result["boxes"])):
        label_index = int(label)
        if label_index >= len(label_to_category):
            raise ValueError(
                f"Model predicted label {label_index}, but only {len(label_to_category)} "
                "category ids are configured")
        category_id = label_to_category[label_index]
        x1, y1, x2, y2 = (float(value) for value in box)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        bbox_record = {
            "image_id": int(image_id), "category_id": category_id,
            "bbox": bbox, "score": float(score),
        }
        bbox_records.append(bbox_record)
        detection = dict(bbox_record)

        if keypoints is not None:
            scores = (keypoint_scores[index] if keypoint_scores is not None
                      else torch.ones(len(keypoints[index]), device=keypoints.device))
            flat = []
            for point, point_score in zip(keypoints[index], scores):
                x, y = (float(value) for value in point)
                # COCO result files accept a confidence in the third slot.
                flat.extend([x, y, float(point_score)])
            keypoint_record = {
                "image_id": int(image_id), "category_id": category_id,
                "keypoints": flat, "score": float(score),
            }
            keypoint_records.append(keypoint_record)
            detection["keypoints"] = flat
            detection["num_keypoints"] = sum(
                float(value) >= keypoint_threshold for value in scores)
        detections.append(detection)
    return detections, bbox_records, keypoint_records


def draw_predictions(image, result, label_to_category, keypoint_threshold,
                     show_labels, show_keypoint_indices):
    draw = ImageDraw.Draw(image)
    keypoints = result.get("keypoints")
    keypoint_scores = result.get("keypoint_scores")
    for index, (score, label, box) in enumerate(zip(
            result["scores"], result["labels"], result["boxes"])):
        x1, y1, x2, y2 = (float(value) for value in box)
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        if show_labels:
            category_id = label_to_category[int(label)]
            draw.text((x1 + 2, max(0, y1 - 12)), f"cat={category_id} {float(score):.3f}", fill="lime")
        if keypoints is None:
            continue
        scores = (keypoint_scores[index] if keypoint_scores is not None
                  else torch.ones(len(keypoints[index])))
        for point_index, (point, point_score) in enumerate(zip(keypoints[index], scores)):
            if float(point_score) < keypoint_threshold:
                continue
            x, y = (float(value) for value in point)
            if not (0 <= x < image.width and 0 <= y < image.height):
                continue
            radius = max(2, round(min(image.size) / 400))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
            if show_keypoint_indices:
                draw.text((x + radius + 1, y + radius + 1), str(point_index), fill="yellow")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True, help="COCO validation/test annotation JSON")
    parser.add_argument("--image-root", required=True, help="Root joined with each COCO file_name")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-size", type=parse_size,
                        help="HEIGHTxWIDTH; defaults to config eval_spatial_size")
    parser.add_argument("--visualization-resolution", choices=("original", "inference"),
                        default="original")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--keypoint-threshold", type=float, default=0.5)
    parser.add_argument("--category-ids", help="Comma-separated COCO ids in model-label order")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--filename-digits", type=int, default=5)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--show-keypoint-indices", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.score_threshold <= 1 or not 0 <= args.keypoint_threshold <= 1:
        parser.error("score and keypoint thresholds must be in [0, 1]")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")

    annotation_path = Path(args.ann_file)
    image_root = Path(args.image_root)
    save_dir = Path(args.save_dir)
    image_save_dir = save_dir / "images"
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_images:
        image_save_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_infos = coco.get("images", [])
    if args.max_images is not None:
        image_infos = image_infos[:args.max_images]
    label_to_category = category_mapping(coco, args.category_ids)

    cfg = YAMLConfig(args.config)
    configured_size = cfg.yaml_cfg.get("eval_spatial_size")
    if args.inference_size is None and configured_size is None:
        parser.error("--inference-size is required when the config has no eval_spatial_size")
    inference_size = args.inference_size or tuple(int(value) for value in configured_size)
    if len(inference_size) != 2:
        raise ValueError("eval_spatial_size must contain [height, width]")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    load_checkpoint(model, args.checkpoint, device)

    dataset_results = []
    bbox_results = []
    keypoint_results = []
    timings = []
    missing_images = []

    for offset, image_info in enumerate(image_infos):
        image_path = image_root / image_info["file_name"]
        if not image_path.is_file():
            missing_images.append(str(image_path))
            print(f"Warning: missing image {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        tensor, inference_image = preprocess(image, inference_size)
        tensor = tensor.unsqueeze(0).to(device)

        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(tensor)
            original_size = torch.tensor(
                [[original_width, original_height]], dtype=torch.float32, device=device)
            original_result = postprocessor(outputs, original_size)[0]
            if args.visualization_resolution == "inference":
                inference_height, inference_width = inference_size
                inference_target = torch.tensor(
                    [[inference_width, inference_height]], dtype=torch.float32, device=device)
                visualization_result = postprocessor(outputs, inference_target)[0]
            else:
                visualization_result = original_result
        timings.append(time.perf_counter() - started)

        original_result = selected_predictions(original_result, args.score_threshold)
        visualization_result = selected_predictions(visualization_result, args.score_threshold)
        detections, boxes, poses = coco_records(
            original_result, image_info["id"], label_to_category, args.keypoint_threshold)
        bbox_results.extend(boxes)
        keypoint_results.extend(poses)

        # Use the number of successfully processed images so missing source
        # files never create gaps in the sequential visualization names.
        output_index = args.start_index + len(dataset_results)
        output_name = f"{output_index:0{args.filename_digits}d}.jpg"
        dataset_results.append({
            "image_id": int(image_info["id"]),
            "file_name": image_info["file_name"],
            "saved_image": None if args.no_images else f"images/{output_name}",
            "original_size": [original_height, original_width],
            "inference_size": list(inference_size),
            "detections": detections,
        })

        if not args.no_images:
            canvas = image.copy() if args.visualization_resolution == "original" else inference_image.copy()
            draw_predictions(canvas, visualization_result, label_to_category,
                             args.keypoint_threshold, args.show_labels,
                             args.show_keypoint_indices).save(image_save_dir / output_name, quality=95)
        print(f"[{offset + 1}/{len(image_infos)}] {image_path.name}: {len(detections)} detections")

    (save_dir / "predictions.json").write_text(
        json.dumps(dataset_results, indent=2), encoding="utf-8")
    (save_dir / "coco_bbox_predictions.json").write_text(
        json.dumps(bbox_results, indent=2), encoding="utf-8")
    (save_dir / "coco_keypoint_predictions.json").write_text(
        json.dumps(keypoint_results, indent=2), encoding="utf-8")
    summary = {
        "images_requested": len(image_infos),
        "images_processed": len(dataset_results),
        "missing_images": missing_images,
        "detections": len(bbox_results),
        "inference_size": list(inference_size),
        "visualization_resolution": args.visualization_resolution,
        "mean_seconds_per_image": sum(timings) / len(timings) if timings else None,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
