"""Hydra entrypoint: collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import json
import math
import os
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from supermimic.config import (
    TrainConfig,
    register_configs,
    save_train_config,
    validate_resume_config,
    validate_train_config,
)
from supermimic.logger import (
    TRACKING_METRICS,
    TRAIN_METRICS,
    TrackingMetrics,
    create_training_logger,
    group_training_metrics,
    prepare_training_log,
)

register_configs()


def train(cfg: DictConfig, logger=None) -> None:
    # Plain dataclasses keep OmegaConf lookups outside torch.compile graphs.
    args = OmegaConf.to_object(cfg)
    import torch
    from torch import nn

    from supermimic import networks
    from supermimic.buffer import REPLAY_VERSION, ReplayBuffer
    from supermimic.checkpoint import initialize_matching_parameters
    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import CRITIC_EXTRA, INV_DIM, JOINT_DIM, MDP, OBS_DIM, normalize_xi, unflatten_state
    from supermimic.motion import MotionLib
    from supermimic.networks import Critic, Policy, WorldModel, grouped_state_error
    from supermimic.robot import default_joint_pos_list

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = args.device
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        validate_resume_config(cfg, ck["args"])
        if args.iterations <= ck["iteration"]:
            raise ValueError(f"iterations must exceed resumed iteration {ck['iteration']}")
    torch.cuda.set_device(dev)
    os.makedirs(args.log_dir, exist_ok=True)
    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(args.motion_dir, kin, dev, clips=args.motion_clips)
    if not getattr(motion, "is_hoi", False):
        raise ValueError("HOI training requires the raw split HOI motions")
    if int(motion.clip_len.min()) <= args.horizon:
        raise ValueError("every HOI clip must be longer than the imagination horizon")
    if ck and ck["args"].get("motion_signature") != motion.signature:
        raise ValueError("resume motion/asset signature differs; use init_from to start a new HOI run")
    if ck and ck.get("replay_version") != REPLAY_VERSION:
        raise ValueError("resume checkpoint has an incompatible replay schema")
    cfg.motion_signature = motion.signature
    config_data = OmegaConf.to_container(cfg, resolve=True)
    with open(os.path.join(args.log_dir, "args.json"), "w") as stream:
        json.dump(config_data, stream, indent=2)
    with open(os.path.join(args.log_dir, "motion_validation.json"), "w") as stream:
        json.dump(motion.validation, stream, indent=2)
    save_train_config(cfg, os.path.join(args.log_dir, "config.yaml"))
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), hand_cfg=args.hand, object_cfg=args.object)
    K, Hh, HZ = args.goal_frames, args.hist, args.horizon
    GOAL_DIM = motion.goal_frame_dim * K
    CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM

    wm = WorldModel(hidden=args.network.wm_hidden).to(dev)
    pi = Policy(
        OBS_DIM,
        GOAL_DIM,
        hist=Hh,
        z_dim=args.network.latent_dim,
        encoder_hidden=args.network.encoder_hidden,
        decoder_hidden=args.network.decoder_hidden,
    ).to(dev)
    critic = Critic(CRITIC_IN, hidden=args.network.critic_hidden).to(dev)
    opt_cfg = args.optimizer
    adam_kwargs = {"lr": args.lr, "eps": opt_cfg.eps, "weight_decay": opt_cfg.weight_decay}
    opt_wm = torch.optim.Adam(wm.parameters(), betas=tuple(opt_cfg.wm_betas), **adam_kwargs)
    opt_pi = torch.optim.Adam(pi.parameters(), betas=tuple(opt_cfg.pi_betas), **adam_kwargs)
    opt_v = torch.optim.Adam(critic.parameters(), betas=tuple(opt_cfg.critic_betas), **adam_kwargs)
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    if ck is not None:
        wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
        opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
        # Restore Adam moments while keeping the active Hydra hyperparameters.
        for opt, betas in ((opt_wm, opt_cfg.wm_betas), (opt_pi, opt_cfg.pi_betas), (opt_v, opt_cfg.critic_betas)):
            for group in opt.param_groups:
                group.update(betas=tuple(betas), eps=opt_cfg.eps, weight_decay=opt_cfg.weight_decay)
        start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    elif args.init_from:
        source = torch.load(args.init_from, map_location="cpu")
        if source["args"].get("robot_model") != "g1_inspire" or source["args"].get("control_version") not in (2, 3):
            raise ValueError("init_from requires a 41-action G1 Inspire checkpoint")
        for name, module in (("wm", wm), ("pi", pi)):
            report = initialize_matching_parameters(module, source[name])
            print(f"[init] {name}: copied {len(report['copied'])} parameter tensors; "
                  f"{len(report['skipped'])} resized/new tensors", flush=True)

    # per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
    ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                           torch.full((JOINT_DIM,), 3e-3), torch.full((JOINT_DIM,), 5e-2),
                           torch.tensor([3e-3]*3 + [5e-3]*3 + [2e-2]*3 + [5e-2]*3))).to(dev)  # fmt: skip

    def _wm_rollout(s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(args.wm_window):
            S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(len(S)))  # (B, k, ...)

    def _wm_err(S_hat, s1, mask):
        std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
        e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
        body_e, hand_e, object_e = grouped_state_error(e)
        e = body_e + args.hand.wm_loss_weight * hand_e + args.object.wm_loss_weight * object_e
        return (e * mask).sum() / s1.shape[0], e[:, -1], body_e[:, -1], object_e[:, -1]

    def _imag_step(S, o_hist, f, xi, xi_n, g):
        a = pi(o_hist, g)
        S1 = wm.step(S, a, xi_n)
        f1 = f + 1
        o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, motion.goal(f1, K)

    def _reward_term(S, f, a, a_prev):
        return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)

    def _critic_input(S, f, xi_n, g):
        return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)

    compile_fn = (lambda fn: fn) if args.no_compile else torch.compile
    wm_rollout = compile_fn(_wm_rollout)
    wm_err = compile_fn(_wm_err)
    imag_step = compile_fn(_imag_step)
    reward_term = compile_fn(_reward_term)
    critic_input_c = compile_fn(_critic_input)

    def lr_at(it: int) -> float:
        if it >= args.lr_decay_iters:
            return args.lr * opt_cfg.lr_final_factor
        floor = opt_cfg.lr_final_factor
        return args.lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))

    # ------------------------------------------------------------------------------------------------
    def wm_update():
        w = buffer.sample_windows(args.wm_batch, args.wm_window)
        S_hat = wm_rollout(w["s"], w["a"], w["xi"])
        loss, e_last, body_last, object_last = wm_err(S_hat, w["s1"], w["mask"])
        opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), opt_cfg.wm_grad_norm)
        opt_wm.step()
        m = w["mask"][:, -1]
        end_error = lambda x: (x.detach() * m).sum() / m.sum().clamp(min=1)
        f_last = w["f"][:, -1]
        f1_last = torch.minimum(f_last + 1, motion.clip_end[motion.clip_id[f_last]])
        contact_mask = m * (motion.contact_label[f1_last] == 1).any(-1)
        contact_error = (object_last.detach() * contact_mask).sum() / contact_mask.sum().clamp_min(1)
        contact_error = torch.where(contact_mask.sum() > 0, contact_error, float("nan"))
        return (loss.detach() / args.wm_window, end_error(e_last), end_error(body_last),
                end_error(object_last), contact_error)

    def sample_valid_starts(B: int):
        starts, hists, n, attempts = [], [], 0, 0
        while n < B:
            attempts += 1
            if attempts > 64:
                raise RuntimeError("Replay has too few frames before the clip end for imagination; "
                                   "collect more transitions or reduce horizon")
            s, h = buffer.sample_starts(2 * B, Hh)
            f = s["f"]
            ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}
        return cat(starts), cat(hists)

    PROFILE = args.profile

    def policy_update():
        if PROFILE:
            torch.cuda.synchronize(dev)
            tq0 = time.time()
        s, h = sample_valid_starts(args.pi_batch)
        S = unflatten_state(s["s"])
        xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = mdp.add_noise(o_hist)
        wm.requires_grad_(False)
        critic.requires_grad_(False)
        B = args.pi_batch
        g0 = motion.goal(f, K)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for j in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = imag_step(S, o_hist, f, xi, xi_n, g)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(len(S)))
        f_all = flat(Fs)
        r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not args.no_critic:
            J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g))
        loss = -J.mean()
        if PROFILE:
            torch.cuda.synchronize(dev)
            tq1 = time.time()
        opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if PROFILE:
            torch.cuda.synchronize(dev)
            tq2 = time.time()
        gn = nn.utils.clip_grad_norm_(pi.parameters(), opt_cfg.pi_grad_norm)
        nn.utils.clip_grad_value_(pi.parameters(), opt_cfg.pi_grad_value)
        if torch.isfinite(gn):
            opt_pi.step()
        wm.requires_grad_(True)
        critic.requires_grad_(True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros(())
        if not args.no_critic:
            with torch.no_grad():
                S_c = tuple(
                    torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1)
                    for i in range(len(S))
                )
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            critic.norm.update(X)
            with torch.no_grad():
                V = critic(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])  # (H, B)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // args.critic_minibatches
            for _ in range(args.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), opt_cfg.critic_grad_norm)
                    opt_v.step()
        if PROFILE:
            torch.cuda.synchronize(dev)
            print(
                f"[prof] imagine fwd {1e3 * (tq1 - tq0):.0f} ms | bwd {1e3 * (tq2 - tq1):.0f} ms | critic {1e3 * (time.time() - tq2):.0f} ms",
                flush=True,
            )
        return loss.detach(), v_loss.detach(), (r_sum / HZ / mdp.max_step_reward).mean(), gn.detach(), c.mean().detach()

    # ------------------------------------------------------------------------------------------------
    sim_cfg = make_sim(dev, args)
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
    obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        nonlocal obs_hist, sim_steps
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            obs_hist = torch.where(
                fresh[:, None, None],
                o.unsqueeze(1).expand_as(obs_hist),
                torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1),
            )
            g = motion.goal(env.f, K)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(obs_hist, g)
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
            out = env.step(a)
            tracking_metrics.update(out)
            buffer.add(out)
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            stats["r"].append(out["r"].mean() / mdp.max_step_reward)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())

    tracking_metrics = TrackingMetrics(dev, mdp.max_step_reward)
    stats = {"r": [], "ep_len": [], "ep_ret": []}
    collect(args.warmup_steps, stats)
    log_path = os.path.join(args.log_dir, "train_log.csv")
    log_columns = prepare_training_log(log_path)
    with open(log_path, "a") as log_f:
        t_start = time.time() - wall_offset
        t_log = time.time()
        step_acc = 0.0
        agg = {"wm": [], "wm8": [], "wm_body": [], "wm_object": [], "wm_object_contact": [],
               "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        for it in range(start_it, args.iterations):
            lr = lr_at(it)
            for o in (opt_wm, opt_pi, opt_v):
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            step_acc += 1.0 / args.utd
            n = int(step_acc)
            step_acc -= n
            prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
            if prof:
                torch.cuda.synchronize(dev)
                tp0 = time.time()
            collect(n, stats)
            if prof:
                torch.cuda.synchronize(dev)
                tp1 = time.time()
            for _ in range(args.wm_batches):
                l, e8, body_e8, object_e8, object_contact_e8 = wm_update()
            if prof:
                torch.cuda.synchronize(dev)
                tp2 = time.time()
            agg["wm"].append(l), agg["wm8"].append(e8)
            agg["wm_body"].append(body_e8)
            agg["wm_object"].append(object_e8)
            if torch.isfinite(object_contact_e8):
                agg["wm_object_contact"].append(object_contact_e8)
            if it >= args.wm_only_iters:
                pl, vl, ir, gn, surv = policy_update()
                if prof:
                    torch.cuda.synchronize(dev)
                    tp3 = time.time()
                    print(
                        f"[prof] collect {1e3 * (tp1 - tp0):.0f} ms ({n} steps) | wm {1e3 * (tp2 - tp1):.0f} ms | policy+critic {1e3 * (tp3 - tp2):.0f} ms",
                        flush=True,
                    )
                (
                    agg["pi"].append(pl),
                    agg["ir"].append(ir),
                    agg["gn"].append(gn),
                    agg["surv"].append(surv),
                )
                if not args.no_critic:
                    agg["v"].append(vl)
            if (it + 1) % args.log_every == 0:
                mean = lambda x: sum(float(v) for v in x) / len(x) if x else math.nan
                ips = args.log_every / (time.time() - t_log)
                t_log = time.time()
                wall_h = (time.time() - t_start) / 3600
                diagnostics = tracking_metrics.summary()
                term_frac = diagnostics.get("term_frac", math.nan)
                row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                       mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr,
                       mean(agg["wm_body"])]  # fmt: skip
                row.extend(diagnostics.get(name, math.nan) for name in TRACKING_METRICS)
                row.append(mean(agg["wm_object"]))
                row.append(mean(agg["wm_object_contact"]))
                metrics = dict(zip(TRAIN_METRICS, row, strict=True))
                log_f.write(",".join(
                    (f"{x:.6g}" if isinstance(x, float) else str(x)) if math.isfinite(x) else ""
                    for x in (metrics.get(name, math.nan) for name in log_columns)
                ) + "\n")
                log_f.flush()
                if logger is not None:
                    logger.log(group_training_metrics(metrics), step=it + 1)
                display = lambda value, spec: format(value, spec) if math.isfinite(value) else "--"
                print(
                    f"it {it + 1} | {wall_h:.3f} h | wm {display(row[3], '.4f')} err {display(row[4], '.4f')} "
                    f"| pi {display(row[5], '.3f')} v {display(row[6], '.4f')} "
                    f"imag_r {display(row[7], '.3f')} surv {display(row[9], '.3f')} "
                    f"| sim_r {display(row[10], '.3f')} ep_len {display(row[11], '.1f')} "
                    f"ep_ret {display(row[12], '.3f')} term {display(row[13], '.2f')} "
                    f"| {ips:.2f} it/s",
                    flush=True,
                )
                stats = {k: [] for k in stats}
                tracking_metrics.reset()
                agg = {k: [] for k in agg}
            if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
                ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                      "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                      "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": config_data,
                      "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "replay_version": REPLAY_VERSION}  # fmt: skip
                last = os.path.join(args.log_dir, "last.pt")
                torch.save(ck, last + ".tmp")
                os.replace(last + ".tmp", last)
                small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at", "replay_version")}
                path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
                torch.save(small, path + ".tmp")
                os.replace(path + ".tmp", path)


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    # Saved resolved snapshots omit Hydra's defaults list; reapply the type schema.
    cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg)
    OmegaConf.resolve(cfg)
    # Hydra's output directory also isolates checkpoints in multirun jobs.
    cfg.log_dir = HydraConfig.get().runtime.output_dir
    cfg.motion_dir = to_absolute_path(cfg.motion_dir)
    if cfg.resume:
        cfg.resume = to_absolute_path(cfg.resume)
    if cfg.init_from:
        cfg.init_from = to_absolute_path(cfg.init_from)
    validate_train_config(cfg)
    with create_training_logger(cfg, OmegaConf.to_container(cfg, resolve=True)) as logger:
        train(cfg, logger)


if __name__ == "__main__":
    main()
