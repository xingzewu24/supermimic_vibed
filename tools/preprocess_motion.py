"""Convert each raw G1 CSV to a same-name, preprocessed 50 Hz NPZ."""

import argparse
import time
from pathlib import Path

import torch

from supermimic.kinematics import G1Kinematics
from supermimic.motion import preprocess_directory
from supermimic.robot import LAFAN_MOTION_DIR, RAW_MOTION_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir", type=Path, default=Path(RAW_MOTION_DIR), help="directory containing raw CSVs"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path(LAFAN_MOTION_DIR), help="directory for processed LAFAN NPZs"
    )
    parser.add_argument("--device", default="cuda:0", help="preprocessing device; cpu is also supported")
    parser.add_argument("--src_fps", type=float, default=30.0)
    parser.add_argument("--rebuild", action="store_true", help="force regeneration of existing NPZs")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    kin = G1Kinematics(str(device))
    start = time.perf_counter()
    paths = preprocess_directory(
        args.source_dir, args.output_dir, kin, str(device), src_fps=args.src_fps, rebuild=args.rebuild
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    size = sum(path.stat().st_size for path in paths)
    print(f"Ready: {len(paths)} NPZ files, {size / 2**20:.1f} MiB; {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
