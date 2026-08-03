import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TVF

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import engine  # noqa: F401
from engine.core import YAMLConfig

WARNING = "KEYPOINT HEAD IS RANDOMLY INITIALIZED - VISUALIZATION IS NOT A VALID POSE PREDICTION"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path):
    path = Path(path)
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in IMAGE_EXTS:
                yield child
    else:
        yield path


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    bad_missing = [k for k in result.missing_keys if "keypoint" not in k]
    if unexpected or bad_missing:
        print("Missing keys:", result.missing_keys)
        print("Unexpected keys:", result.unexpected_keys)
        raise RuntimeError("Checkpoint is not compatible with ECDetPose.")
    return checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}


def preprocess(image, eval_spatial_size):
    if eval_spatial_size is not None:
        height, width = [int(v) for v in eval_spatial_size]
        image = image.resize((width, height))
    tensor = TVF.to_tensor(image)
    tensor = TVF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor


def draw_prediction(image, result, score_threshold, keypoint_threshold, show_labels, show_keypoint_indices):
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 22], fill="black")
    draw.text((4, 4), WARNING, fill="yellow")
    for score, label, box, keypoints, kp_scores in zip(
        result["scores"], result["labels"], result["boxes"], result.get("keypoints", []), result.get("keypoint_scores", [])
    ):
        if float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        if show_labels:
            draw.text((x1, max(24, y1 - 12)), f"{int(label)} {float(score):.2f}", fill="lime")
        for idx, (pt, kp_score) in enumerate(zip(keypoints, kp_scores)):
            if float(kp_score) < keypoint_threshold:
                continue
            x, y = [float(v) for v in pt]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="red")
            if show_keypoint_indices:
                draw.text((x + 3, y + 3), str(idx), fill="white")
    return image


def to_jsonable(result):
    return {k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--keypoint-threshold", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--show-keypoint-indices", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    meta = load_checkpoint(model, args.checkpoint, device)
    print(WARNING)
    if meta:
        print(json.dumps(meta, indent=2, default=str))

    raw_predictions = {}
    coco_bbox = []
    coco_keypoints = []
    timing = {}

    for image_path in iter_images(args.input):
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess(image, cfg.yaml_cfg.get("eval_spatial_size")).unsqueeze(0).to(device)
        sizes = torch.tensor([[image.width, image.height]], dtype=torch.float32, device=device)
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(tensor)
            results = postprocessor(outputs, sizes)
        elapsed = time.perf_counter() - start
        result = results[0]
        timing[str(image_path)] = elapsed

        if "keypoints" in result:
            kpts = result["keypoints"]
            boxes = result["boxes"]
            inside = (
                (kpts[..., 0] >= boxes[:, None, 0] - 1e-4)
                & (kpts[..., 0] <= boxes[:, None, 2] + 1e-4)
                & (kpts[..., 1] >= boxes[:, None, 1] - 1e-4)
                & (kpts[..., 1] <= boxes[:, None, 3] + 1e-4)
            )
            print(f"{image_path.name}: {inside.float().mean().item() * 100:.2f}% keypoints inside boxes")

        drawn = draw_prediction(image.copy(), result, args.score_threshold, args.keypoint_threshold, args.show_labels, args.show_keypoint_indices)
        drawn.save(output_dir / image_path.name)
        raw_predictions[str(image_path)] = to_jsonable(result)

        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            coco_bbox.append({"image_id": image_path.stem, "category_id": int(label), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
        if "keypoints" in result:
            for score, label, keypoints, kp_scores in zip(result["scores"], result["labels"], result["keypoints"], result["keypoint_scores"]):
                flat = []
                for (x, y), kp_score in zip(keypoints, kp_scores):
                    flat.extend([float(x), float(y), 2 if float(kp_score) >= args.keypoint_threshold else 1])
                coco_keypoints.append({"image_id": image_path.stem, "category_id": int(label), "keypoints": flat, "score": float(score)})

    if args.save_json:
        (output_dir / "raw_predictions.json").write_text(json.dumps(raw_predictions, indent=2), encoding="utf-8")
        (output_dir / "coco_bbox_predictions.json").write_text(json.dumps(coco_bbox, indent=2), encoding="utf-8")
        (output_dir / "coco_keypoint_predictions.json").write_text(json.dumps(coco_keypoints, indent=2), encoding="utf-8")
        (output_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
