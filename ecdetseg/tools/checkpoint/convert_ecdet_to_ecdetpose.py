import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import engine  # noqa: F401
from engine.core import YAMLConfig


def unwrap_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "ema", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecdet-config", required=True)
    parser.add_argument("--ecdetpose-config", required=True)
    parser.add_argument("--ecdet-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-keypoints", type=int, default=31)
    args = parser.parse_args()

    cfg = YAMLConfig(args.ecdetpose_config, num_keypoints=args.num_keypoints)
    model = cfg.model
    checkpoint = torch.load(args.ecdet_checkpoint, map_location="cpu")
    state = unwrap_state(checkpoint)
    load_result = model.load_state_dict(state, strict=False)

    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    allowed_missing = [k for k in missing if "keypoint" in k]
    bad_missing = sorted(set(missing) - set(allowed_missing))
    if bad_missing or unexpected:
        print("Missing keys:", json.dumps(missing, indent=2))
        print("Unexpected keys:", json.dumps(unexpected, indent=2))
        raise SystemExit("Refusing conversion because non-keypoint detection weights did not load strictly.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "meta": {
            "model_family": "ECDetPose",
            "variant": Path(args.ecdetpose_config).stem.split("_")[-1].upper(),
            "num_keypoints": args.num_keypoints,
            "detection_initialized_from": args.ecdet_checkpoint,
            "keypoint_head_initialized": "random",
            "git_commit": os.popen("git rev-parse HEAD").read().strip(),
            "config": args.ecdetpose_config,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },
    }
    torch.save(payload, output)
    print(json.dumps({"output": str(output), "sha256": sha256(output), "missing_keys": missing}, indent=2))


if __name__ == "__main__":
    main()
