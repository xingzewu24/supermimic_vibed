"""Assert state round-trip and FK agreement, then exercise motion playback."""

import argparse

import torch
from cli import add_sim_args

from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, body_state
from supermimic.motion import MotionLib
from supermimic.robot import MOTION_DIR, TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import quat_to_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--motion_dir", default=MOTION_DIR)
    add_sim_args(parser)
    args = parser.parse_args()
    dev = args.device
    torch.cuda.set_device(dev)
    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(args.motion_dir, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    env = TrackingSim(make_sim(dev, args), args.num_envs, motion, mdp, eval_mode=True)
    ids = torch.arange(args.num_envs, device=dev)
    clips = ids % len(motion.names)
    frames = torch.minimum(motion.clip_start[clips] + 100, motion.clip_end[clips] - 1)
    env.reset(ids, frames)
    state = env.state()
    for name, actual, expected in zip(("p", "R", "v", "w", "q", "qd", "op", "oR", "ov", "ow"), state, motion.ref_state(frames)):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        print(f"roundtrip {name}: max err {(actual - expected).abs().max().item():.2e}")
    body_ids = [env.robot.body_names.index(name) for name in TRACKED_BODIES]
    pose = env.robot.data.body_link_pose_w[:, body_ids]
    vel = env.robot.data.body_link_vel_w[:, body_ids]
    bp, bR, bv, bw = kin.forward(*body_state(state))
    for name, actual, expected in (
        ("position", pose[..., :3], bp),
        ("rotation", quat_to_mat(pose[..., 3:7]), bR),
        ("linear velocity", vel[..., :3], bv),
        ("angular velocity", vel[..., 3:], bw),
    ):
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
        print(f"FK {name}: max err {(actual - expected).abs().max().item():.2e}")
    start_time = env.sim.data.time.clone()
    for _ in range(args.steps):
        f1 = torch.minimum(env.f + 1, motion.clip_end[motion.clip_id[env.f]])
        action = (motion.q[f1] - env.default_q) / env.alpha
        hand_action = torch.atanh(((motion.hand_reference(f1) - env.hand.mid) / env.hand.half).clamp(-0.999, 0.999))
        action = torch.cat((action, hand_action), -1)
        out = env.step(action)
        assert torch.isfinite(out["s1"]).all() and torch.isfinite(out["r"]).all()
    torch.testing.assert_close(
        env.sim.data.time - start_time, torch.full_like(start_time, args.steps * 0.02), atol=1e-5, rtol=1e-5
    )
    print(f"PASS: {args.steps} control steps, four 5 ms substeps each")


if __name__ == "__main__":
    main()
