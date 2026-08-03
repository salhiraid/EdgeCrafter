import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine  # noqa: F401
from engine.core import YAMLConfig


def count_params(module, trainable=None):
    params = module.parameters()
    if trainable is None:
        return sum(p.numel() for p in params)
    return sum(p.numel() for p in params if p.requires_grad == trainable)


def bucket_params(model):
    buckets = {
        "backbone_parameters": 0,
        "encoder_parameters": 0,
        "decoder_parameters": 0,
        "detection_head_parameters": 0,
        "keypoint_head_parameters": 0,
    }
    for name, param in model.named_parameters():
        n = param.numel()
        if ".backbone." in name or name.startswith("backbone."):
            buckets["backbone_parameters"] += n
        elif ".encoder." in name or name.startswith("encoder."):
            buckets["encoder_parameters"] += n
        elif "keypoint" in name:
            buckets["keypoint_head_parameters"] += n
        elif any(x in name for x in ("score_head", "bbox_head", "enc_score_head", "enc_bbox_head", "pre_bbox_head")):
            buckets["detection_head_parameters"] += n
        elif ".decoder." in name or name.startswith("decoder."):
            buckets["decoder_parameters"] += n
    return buckets


def flops_if_available(model, input_shape):
    try:
        from calflops import calculate_flops
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    flops, macs, params = calculate_flops(model=model, input_shape=input_shape, output_as_string=True)
    return {"available": True, "flops": flops, "macs": macs, "params": params}


def report(config, checkpoint=None, device="cpu"):
    cfg = YAMLConfig(config, ViTAdapter={"skip_load_backbone": True, "weights_path": None})
    model = cfg.model.to(device).eval()
    info = {
        "config": config,
        "total_parameters": count_params(model),
        "trainable_parameters": count_params(model, trainable=True),
        **bucket_params(model),
        "checkpoint_file_size": None,
        "input_resolution": cfg.yaml_cfg.get("eval_spatial_size"),
        "number_of_queries": getattr(model.decoder, "num_queries", None),
        "number_of_decoder_layers": getattr(model.decoder, "num_layers", None),
        "flops": flops_if_available(model, (1, 3, *cfg.yaml_cfg.get("eval_spatial_size", [640, 640]))),
    }
    if checkpoint:
        path = Path(checkpoint)
        info["checkpoint_file_size"] = path.stat().st_size if path.exists() else None
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows = [report(config, args.checkpoint, args.device) for config in args.configs]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
