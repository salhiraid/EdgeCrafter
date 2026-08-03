#!/usr/bin/env python3
"""Download five images, run one training epoch, and report loss artifacts."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ecvehicle_smoke"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    ecpose_root = Path(__file__).resolve().parents[2]
    output_dir = (ecpose_root / args.output_dir).resolve()
    data_dir = ecpose_root / "smoke_data"

    if not args.skip_download:
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("download_vehicle_smoke_data.py")),
            "--output", str(data_dir),
        ], check=True, cwd=ecpose_root)

    command = [
        sys.executable, "train.py",
        "-c", "configs/ecvehicle/ecvehicle_smoke.yml",
        "--device", args.device,
        "--output-dir", str(output_dir),
        "--summary-dir", str(output_dir / "tensorboard"),
        "--seed", "0",
    ]
    if args.device.startswith("cuda"):
        command.append("--use-amp")

    print("running:", " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(command, check=True, cwd=ecpose_root, env=env)

    log_path = output_dir / "log.txt"
    if not log_path.exists():
        raise RuntimeError(f"Training completed without producing {log_path}")
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    latest = records[-1]
    losses = {key: value for key, value in latest.items() if key.startswith("train_loss")}
    if not losses:
        raise RuntimeError(f"No training losses found in {log_path}")

    event_files = list((output_dir / "tensorboard").glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("No TensorBoard event file was produced")

    print("\nSmoke-test losses:")
    for name, value in sorted(losses.items()):
        print(f"  {name}: {value:.6f}")
    print(f"\nJSONL metrics: {log_path}")
    print(f"Text log:      {output_dir / 'training.log'}")
    print(f"TensorBoard:   tensorboard --logdir {output_dir / 'tensorboard'}")
    print(f"Checkpoint:    {output_dir / 'checkpoint.pth'}")


if __name__ == "__main__":
    main()
