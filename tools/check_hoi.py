"""Validate split HOI references without changing data; optionally exercise all resets on GPU."""

import argparse
import json
from pathlib import Path

import torch

from supermimic.config import DEFAULT_EPA_HORIZON, DEFAULT_MOTION_DIR, DEFAULT_NJMAX
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_dir", default=DEFAULT_MOTION_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path, default=Path("/tmp/hoi_validation.json"))
    parser.add_argument("--physics", action="store_true", help="check reset/one-step physics at every frame on CUDA")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--njmax", type=int, default=DEFAULT_NJMAX)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--epa_horizon", type=int, default=DEFAULT_EPA_HORIZON)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("num_envs must be positive")
    kin = G1Kinematics(args.device)
    motion = MotionLib.from_dir(args.motion_dir, kin, args.device)
    report = {**motion.validation, "motion_signature": motion.signature}
    if args.physics:
        if not args.device.startswith("cuda"):
            parser.error("--physics requires --device cuda:0 (or another CUDA device)")
        from supermimic.env import TrackingSim, make_sim
        from supermimic.mdp import MDP, flatten_state
        from supermimic.robot import default_joint_pos_list

        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device))
        env = TrackingSim(make_sim(args.device, args), args.num_envs, motion, mdp, eval_mode=True)
        ids = torch.arange(args.num_envs, device=args.device)
        max_reset_error, min_rows, max_rows = 0.0, float("inf"), 0
        max_reset_penetration, max_step_penetration, worst_frame = 0.0, 0.0, None
        for start in range(0, motion.num_frames, args.num_envs):
            f = (start + ids).clamp_max(motion.num_frames - 1)
            env.reset(ids, f)
            actual, expected = flatten_state(env.state()), flatten_state(motion.ref_state(f))
            # Quaternions have a canonical sign on both sides of this comparison.
            max_reset_error = max(max_reset_error, (actual - expected).abs().max().item())
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
            reference_contacts = env.contact_diagnostics.compute(f)
            depth, local_index = reference_contacts["object_penetration"].max(0)
            if float(depth) > max_reset_penetration:
                max_reset_penetration, worst_frame = float(depth), int(f[local_index])
            transition = env.step(torch.zeros(args.num_envs, 41, device=args.device))
            max_step_penetration = max(max_step_penetration, float(transition["terms"]["object_penetration"].max()))
            # Capacity is checked at every substep by TrackingSim, including contact-buffer overflow flags.
            min_rows = min(min_rows, int(env._nefc.min()))
            max_rows = max(max_rows, int(env._peak_nefc.max()))
        report["physics"] = {"frames": motion.num_frames, "worlds": args.num_envs,
                             "max_reset_component_error": max_reset_error, "max_constraint_rows": max_rows,
                             "min_constraint_rows": min_rows, "njmax": env.cfg.physics.njmax,
                             "nconmax": env.cfg.physics.nconmax,
                             "epa_horizon": env.cfg.epa_horizon,
                             "max_reset_robot_object_penetration_m": max_reset_penetration,
                             "max_one_step_robot_object_penetration_m": max_step_penetration,
                             "worst_reset_penetration_global_frame": worst_frame}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"HOI validation report: {args.report}")


if __name__ == "__main__":
    main()
