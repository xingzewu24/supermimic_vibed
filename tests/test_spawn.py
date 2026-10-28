"""Spawn G1s on mjlab/MuJoCo-Warp and report stepping throughput."""

import argparse
import time

import torch
from cli import add_sim_args
from mjlab.sim import Simulation

from supermimic.env import make_scene, make_sim
from supermimic.robot import ALL_HAND_JOINT_NAMES, G1_JOINT_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    add_sim_args(parser)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    cfg = make_sim(args.device, args)
    scene = make_scene(args.num_envs, args.device, eval_mode=True)
    sim = Simulation(args.num_envs, cfg.physics, model=scene.compile(), device=args.device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    assert set(robot.joint_names) == set(G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES)
    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    sim.forward()
    print("JOINTS", robot.joint_names)
    print("BODIES", robot.body_names)
    print("mass", sim.mj_model.body_mass.sum(), "CUDA graphs", sim.use_cuda_graph)
    for _ in range(3):
        torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        for _ in range(args.steps):
            for _ in range(cfg.decimation):
                scene.write_data_to_sim()
                sim.step()
                scene.update(cfg.physics.mujoco.timestep)
        torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - start
        assert torch.isfinite(sim.data.qpos).all()
        sim.forward()
        z = robot.data.root_link_pos_w[:, 2].mean().item()
        print(f"steps/s (control 50 Hz) = {args.steps * args.num_envs / elapsed:.0f}; root z mean {z:.3f}")


if __name__ == "__main__":
    main()
