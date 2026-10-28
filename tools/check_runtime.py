"""Verify the locked PyTorch/Warp CUDA 12 runtime on the selected GPU."""

import argparse
from importlib.metadata import version

import torch
import warp as wp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print("mjlab", version("mjlab"), "MuJoCo", version("mujoco"), "MuJoCo-Warp", version("mujoco-warp"))
    print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, "Warp", version("warp-lang"))
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA; check the NVIDIA driver and GPU visibility.")
    torch.cuda.set_device(args.device)
    x = torch.ones(4, device=args.device)
    torch.testing.assert_close(x + x, torch.full_like(x, 2))
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("Warp cannot access CUDA. Recreate the environment with uv sync --locked.")
    y = wp.ones(4, device=device, dtype=wp.float32)
    torch.testing.assert_close(wp.to_torch(y), x)
    print("GPU:", torch.cuda.get_device_name(args.device), "— PyTorch/Warp CUDA checks passed")


if __name__ == "__main__":
    main()
