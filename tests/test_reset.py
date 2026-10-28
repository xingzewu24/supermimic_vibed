"""Assert full and partial resets are immediately visible without stale state."""

import argparse

import torch
from cli import add_sim_args

from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, unflatten_state
from supermimic.motion import MotionLib
from supermimic.robot import MOTION_DIR, default_joint_pos_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--motion_dir", default=MOTION_DIR)
    add_sim_args(parser)
    args = parser.parse_args()
    dev = args.device
    torch.cuda.set_device(dev)
    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(args.motion_dir, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    env = TrackingSim(make_sim(dev, args), args.num_envs, motion, mdp, eval_mode=True)
    action = torch.zeros(args.num_envs, 41, device=dev)
    for _ in range(5):
        env.step(action)
    before = env.state()
    ids = torch.arange(0, args.num_envs, 2, device=dev)
    others = torch.arange(1, args.num_envs, 2, device=dev)
    clips = ids % len(motion.names)
    frames = torch.minimum(motion.clip_start[clips] + 200, motion.clip_end[clips] - 1)
    env.reset(ids, frames)
    ref = motion.ref_state(frames)
    for actual, expected, old in zip(env.state(), ref, before):
        torch.testing.assert_close(actual[ids], expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual[others], old[others], atol=0, rtol=0)
    out = env.step(action)
    for actual, expected in zip(unflatten_state(out["s"]), ref):
        torch.testing.assert_close(actual[ids], expected, atol=2e-5, rtol=2e-5)
    assert (env.ep_step[ids] == 1).all()
    print("PASS: partial reset preserves other environments and the next replay state matches the reference")


if __name__ == "__main__":
    main()
