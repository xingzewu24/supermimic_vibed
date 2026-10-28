"""Shared command-line options for headless mjlab tools."""

from __future__ import annotations

import argparse

from supermimic.config import DEFAULT_EPA_HORIZON, DEFAULT_NJMAX


def add_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0", help="PyTorch/Warp CUDA device, e.g. cuda:1")
    parser.add_argument("--njmax", type=int, default=DEFAULT_NJMAX, help="constraints allocated per environment")
    parser.add_argument("--nconmax", type=int, default=128, help="contacts allocated per environment")
    parser.add_argument("--epa_horizon", type=int, default=DEFAULT_EPA_HORIZON, help="EPA edges per convex pair")
    parser.add_argument(
        "--headless", action="store_true", help="accepted for compatibility; simulation is always headless"
    )
