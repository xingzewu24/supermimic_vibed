"""Hydra checkpoint evaluation and trajectory recording on the motion library.

Every motion starts at its first frame, with friction 1.0 and no domain randomization
or observation/action noise. Use run_dir=logs/<run> watch=true to monitor training.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import time

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from supermimic.checkpoint import load_policy
from supermimic.config import (
    DEFAULT_MOTION_DIR,
    EvalConfig,
    HandControlConfig,
    ObjectTrackingConfig,
    register_configs,
    validate_control_checkpoint,
)

register_configs()


def evaluate_run(args: DictConfig) -> None:
    import numpy as np
    import torch

    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.motion import MotionLib
    from supermimic.robot import ALL_HAND_JOINT_NAMES, default_joint_pos_list

    dev = args.device
    torch.cuda.set_device(dev)
    kin = G1Kinematics(dev)
    env, motion_dir = None, None

    @torch.no_grad()
    def evaluate(ck: dict, record: str | None = None) -> dict:
        nonlocal env, motion_dir
        a_ = ck["args"]
        validate_control_checkpoint(a_)
        selected_dir = args.motion_dir or a_.get("motion_dir", DEFAULT_MOTION_DIR)
        clips = a_.get("motion_clips") if args.motion_dir is None else None
        env_key = (selected_dir, clips, a_["hand"], a_["object"], a_.get("motion_signature"))
        if env is None or env_key != motion_dir:
            motion = MotionLib.from_dir(selected_dir, kin, dev, **({"clips": clips} if clips is not None else {}))
            mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev),
                      hand_cfg=HandControlConfig(**a_["hand"]), object_cfg=ObjectTrackingConfig(**a_["object"]))
            if args.motion_dir is None and a_.get("motion_signature") != motion.signature:
                raise ValueError("Checkpoint motion/asset signature changed; set motion_dir explicitly to evaluate new data")
            sim_cfg = make_sim(dev, args)
            env = TrackingSim(sim_cfg, len(motion.names), motion, mdp, eval_mode=True)
            motion_dir = env_key
        motion, mdp = env.motion, env.mdp
        C = len(motion.names)
        K, Hh = a_["goal_frames"], a_["hist"]
        pi = load_policy(ck, dev)
        ids = torch.arange(C, device=dev)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, motion.clip_start.clone())
        clip_end = motion.clip_end.clone()
        ret = torch.zeros(C, device=dev)
        mae_count = torch.zeros((), dtype=torch.long, device=dev)
        error_names = ("joint_mae", "body_position_error", "body_rotation_error",
                       "object_position_error", "object_rotation_error", "object_linear_velocity_error",
                       "object_angular_velocity_error", "contact_match_rate", "object_penetration",
                       "contact_slip_speed", "grasp_loss")
        sums = {name: torch.zeros((), dtype=torch.float64, device=dev) for name in error_names}
        hist = None
        traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": [], "hand_q": [], "hand_qd": [],
                "hand_reference": [], "action": [], "object_p": [], "object_quat": [],
                "object_v": [], "object_w": []}
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, motion.goal(env.f, K))
            alive = ~env.done_eval.clone()
            if record:
                from supermimic.rotations import mat_to_quat

                traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
                traj["q"].append(S[4][:, :29].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
                traj["hand_q"].append(env.robot.data.joint_pos[:, env.all_hand_joint_ids].cpu())
                traj["hand_qd"].append(env.robot.data.joint_vel[:, env.all_hand_joint_ids].cpu())
                traj["hand_reference"].append(motion.hand_reference(env.f).cpu()), traj["action"].append(a.cpu())
                traj["object_p"].append(S[6].cpu()), traj["object_quat"].append(mat_to_quat(S[7]).cpu())
                traj["object_v"].append(S[8].cpu()), traj["object_w"].append(S[9].cpu())
            out = env.step(a)
            ret += torch.where(alive, out["r"], 0.0)
            # The transition terms already compare S1 to f1 for body and object tracking.
            for name in error_names:
                sums[name] += torch.where(alive, out["terms"][name], 0.0).sum()
            mae_count += alive.sum()
            env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
        norm = ret / ((motion.clip_len - 1).float() * mdp.max_step_reward)
        res = {
            "norm_return": norm.mean().item(),
            "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * mdp.max_step_reward)).item(),
            "n_failed": int(env.failed_eval.sum().item()),
            "failed": [motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
            "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
            "mae": (sums.pop("joint_mae") / mae_count.clamp_min(1)).item(),
            "success_rate": 1.0 - env.failed_eval.float().mean().item(),
            **{key: (value / mae_count.clamp_min(1)).item() for key, value in sums.items()},
        }
        if record:
            np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                                ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                                clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy(),
                                hand_joint_names=np.array(ALL_HAND_JOINT_NAMES),
                                ref_hand_q=motion.hand_q.cpu().numpy(), ref_hand_qd=motion.hand_qd.cpu().numpy(),
                                ref_object_p=motion.object_p.cpu().numpy(), ref_object_R=motion.object_R.cpu().numpy(),
                                ref_object_v=motion.object_v.cpu().numpy(), ref_object_w=motion.object_w.cpu().numpy(),
                                object_names=np.array(motion.object_names), object_path=np.array(motion.object_path),
                                motion_signature=np.array(motion.signature), state_version=np.array(a_["state_version"]),
                                robot_model=np.array("g1_inspire"),
                                model_metadata=np.array(json.dumps(kin.cache_key())))  # fmt: skip
        return res

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        res = evaluate(ck, args.record)
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        return
    out_csv = os.path.join(args.run_dir, "eval.csv")
    metric_columns = ["success_rate", "body_position_error", "body_rotation_error",
                      "object_position_error", "object_rotation_error", "object_linear_velocity_error",
                      "object_angular_velocity_error", "contact_match_rate", "object_penetration",
                      "contact_slip_speed", "grasp_loss", "mae"]
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = list(reader)
        done = {int(row["iteration"]) for row in rows}
        if columns and any(key not in columns for key in metric_columns):
            with open(out_csv, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[*[c for c in columns if c not in metric_columns], *metric_columns])
                writer.writeheader()
                writer.writerows(rows)
    else:
        with open(out_csv, "w") as stream:
            stream.write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed," + ",".join(metric_columns) + "\n")
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
        selected = [c for index, c in enumerate(ckpts, start=1) if index % args.every == 0]
        todo = [c for c in selected if int(os.path.basename(c)[5:11]) not in done]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(ck)
            row = dict(iteration=it, train_wall_h=f"{ck['wall_time'] / 3600:.4f}", saved_at=ck["saved_at"],
                       eval_at=time.strftime("%Y-%m-%d %H:%M:%S"), norm_return=f"{res['norm_return']:.4f}",
                       norm_return_framew=f"{res['norm_return_framew']:.4f}", n_failed=res["n_failed"],
                       failed="|".join(res["failed"]), **{key: f"{res[key]:.6f}" for key in metric_columns})
            with open(out_csv, "a") as stream:
                with open(out_csv) as header:
                    columns = next(csv.reader(header))
                csv.DictWriter(stream, fieldnames=columns).writerow(row)
            with open(c.replace(".pt", "_eval.json"), "w") as stream:
                json.dump(res, stream, indent=1)
            print(
                f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, "
                f"failed {res['n_failed']}/{len(env.motion.names)}, MAE {res['mae']:.4f} rad ({time.time() - t0:.0f}s)",
                flush=True,
            )
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)


@hydra.main(version_base="1.3", config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(EvalConfig), cfg)
    OmegaConf.resolve(cfg)
    for key in ("run_dir", "motion_dir", "ckpt", "record"):
        if cfg[key] is not None:
            cfg[key] = to_absolute_path(cfg[key])
    if cfg.every <= 0:
        raise ValueError("every must be positive")
    if cfg.record and not cfg.ckpt:
        raise ValueError("record requires ckpt=<checkpoint path>")
    evaluate_run(cfg)


if __name__ == "__main__":
    main()
