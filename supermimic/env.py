"""Motion-tracking simulator on mjlab with native MuJoCo-Warp physics.

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`supermimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
import warp as wp
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from .config import DEFAULT_EPA_HORIZON, DEFAULT_NJMAX
from .contact import ContactDiagnostics
from .mdp import ACTION_DIM, BODY_JOINT_DIM, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .physics import configure_epa_horizon, overflow_names
from .robot import (
    ALL_HAND_JOINT_NAMES,
    G1_JOINT_NAMES,
    HAND_JOINT_NAMES,
    HAND_MIMIC_JOINT_NAMES,
    action_scale,
    get_g1_cfg,
    get_object_cfg,
)
from .rotations import mat_to_quat, quat_to_mat


@dataclass
class TrackingSimCfg:
    device: str
    physics: SimulationCfg
    decimation: int = 4
    epa_horizon: int = DEFAULT_EPA_HORIZON


def make_sim(device: str, args=None) -> TrackingSimCfg:
    """50 Hz control with four 5 ms implicit PD physics substeps."""
    return TrackingSimCfg(
        device=device,
        epa_horizon=getattr(args, "epa_horizon", DEFAULT_EPA_HORIZON),
        physics=SimulationCfg(
            njmax=getattr(args, "njmax", DEFAULT_NJMAX),
            nconmax=getattr(args, "nconmax", 128),
            mujoco=MujocoCfg(timestep=CONTROL_DT / 4, cone="pyramidal", integrator="implicitfast"),
        ),
    )


def make_scene(num_envs: int, device: str, eval_mode: bool = False, object_path: str | None = None) -> Scene:
    def add_ground(spec: mujoco.MjSpec):
        spec.worldbody.add_geom(
            name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0.0, 0.0, 0.05), contype=1, conaffinity=0,
            friction=(1.0 if eval_mode else 0.3, 0.005, 0.0001), condim=3,
        )

    entities = {"robot": get_g1_cfg()}
    if object_path is not None:
        entities["object"] = get_object_cfg(object_path)
    return Scene(SceneCfg(
        num_envs=num_envs, env_spacing=0.0,
        entities=entities, spec_fn=add_ground,
    ), device=device)


@requires_model_fields("geom_friction", "body_ipos", recompute=RecomputeLevel.set_const)
def _apply_model_context(sim, robot, xi, torso_id):
    """Keep both randomized properties and their derived constants per world."""
    sim.model.geom_friction[:, robot.indexing.geom_ids, 0] = xi[:, :1]
    torso_global = robot.indexing.body_ids[torso_id]
    sim.model.body_ipos[:, torso_global] += xi[:, 30:33]
    sim.recompute_constants(RecomputeLevel.set_const)


class TrackingSim:
    """Batched G1 motion tracking using mjlab scenes, entities and simulation."""

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = False):
        self.device = sim_cfg.device
        self.cfg = sim_cfg
        if not getattr(motion, "is_hoi", False):
            raise ValueError("TrackingSim requires HOI motions containing full hands and a dynamic object")
        if reset_noise:
            raise ValueError("Independent reset noise is disabled for HOI; joint robot/object reset must preserve contact")
        self.scene = make_scene(num_envs, self.device, eval_mode, motion.object_path)
        # Must precede Simulation.__init__, which allocates collision workspaces during graph capture.
        configure_epa_horizon(sim_cfg.epa_horizon)
        self.sim = Simulation(num_envs, sim_cfg.physics, model=self.scene.compile(), device=self.device)
        # Expand before entity initialization: entities hold views into these arrays.
        if not eval_mode:
            self.sim.expand_model_fields(_apply_model_context.model_fields)
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self._overflow = wp.to_torch(self.sim.wp_data.overflow)
        self._nefc = wp.to_torch(self.sim.wp_data.nefc)
        self._peak_nefc = torch.zeros_like(self._nefc)
        self.robot = self.scene["robot"]
        self.object = self.scene["object"]
        if set(self.robot.joint_names) != set(G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES):
            raise ValueError(f"Unexpected robot joints: {self.robot.joint_names}")
        # The CSV/policy order is independent of the MJCF tree order.
        self.joint_ids = torch.tensor(
            [self.robot.joint_names.index(n) for n in G1_JOINT_NAMES], device=self.device,
        )
        self.N = num_envs
        self.hand_joint_ids = torch.tensor(
            [self.robot.joint_names.index(n) for n in HAND_JOINT_NAMES], device=self.device,
        )
        self.hand_mimic_joint_ids = torch.tensor(
            [self.robot.joint_names.index(n) for n in HAND_MIMIC_JOINT_NAMES], device=self.device,
        )
        self.all_hand_joint_ids = torch.cat((self.hand_joint_ids, self.hand_mimic_joint_ids))
        self.state_joint_ids = torch.cat((self.joint_ids, self.all_hand_joint_ids))
        self.hand = mdp.hand
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.contact_diagnostics = ContactDiagnostics(self)
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)] ----
        self.xi = torch.zeros(self.N, 33, device=dev)
        if eval_mode:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, 30:33] = com
            _apply_model_context(self.sim, self.robot, self.xi, self.torso_id)

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, ACTION_DIM, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = mdp.max_step_reward / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ state
    def _free_state(self, entity):
        pose = self.sim.data.qpos[:, entity.indexing.free_joint_q_adr].clone()
        vel = self.sim.data.qvel[:, entity.indexing.free_joint_v_adr].clone()
        R = quat_to_mat(pose[:, 3:7])
        return (pose[:, :3] - self.scene.env_origins, R, vel[:, :3],
                (R @ vel[:, 3:6].unsqueeze(-1)).squeeze(-1))

    def state(self):
        # mj_step integrates qpos/qvel after computing xpos/cvel. Read the
        # integrated state directly so reward and replay refer to the same time.
        d = self.robot.data
        return (*self._free_state(self.robot), d.joint_pos[:, self.state_joint_ids].clone(),
                d.joint_vel[:, self.state_joint_ids].clone(), *self._free_state(self.object))

    def _fail(self, bad: torch.Tensor, message: str):
        ids = bad.nonzero(as_tuple=False).flatten()[:8]
        frames = self.f[ids]
        clips = [self.motion.names[i] for i in self.motion.clip_id[frames].cpu().tolist()]
        raise RuntimeError(
            f"{message}; env_ids={ids.cpu().tolist()}, frames={frames.cpu().tolist()}, clips={clips}"
        )

    def _ensure_finite(self, value: torch.Tensor, label: str):
        bad = ~torch.isfinite(value).reshape(self.N, -1).all(-1)
        if bool(bad.any()):
            self._fail(bad, f"Non-finite {label}")

    def _check_physics(self, stage: str):
        """Stop before invalid transitions reach reward, return statistics or replay."""
        torch.maximum(self._peak_nefc, self._nefc, out=self._peak_nefc)
        capacity = self.sim.wp_data.njmax
        overflow = (self._overflow != 0) | (self._peak_nefc > capacity)
        nonfinite = torch.zeros_like(overflow)
        for name in ("qpos", "qvel", "qacc", "qfrc_constraint"):
            nonfinite |= ~torch.isfinite(getattr(self.sim.data, name)).all(-1)
        bad = overflow | nonfinite
        if not bool(bad.any()):
            return
        reasons = []
        if bool(overflow.any()):
            from mujoco_warp._src.types import OverflowType

            required = int(self._peak_nefc.max())
            masks = self._overflow[bad][:8].cpu().tolist()
            mask_union = 0
            for mask in masks:
                mask_union |= mask
            hints = []
            if mask_union & int(OverflowType.EPA_HORIZON):
                epa = getattr(getattr(self, "cfg", None), "epa_horizon", DEFAULT_EPA_HORIZON)
                hints.append(f"EPA horizon edge buffer overflow (epa_horizon={epa}); increase epa_horizon; "
                             "njmax/nconmax do not size this buffer")
            if required > capacity or mask_union & int(OverflowType.NEFC | OverflowType.NJMAX_NNZ):
                hints.append("increase njmax for constraint/Jacobian capacity")
            if mask_union & int(OverflowType.BROADPHASE | OverflowType.NARROWPHASE | OverflowType.CCD):
                hints.append("inspect contact/collision pair capacity and nconmax")
            reasons.append(
                f"simulation capacity overflow (njmax={capacity}, observed nefc={required}, "
                f"overflow flags={masks}, overflow types={[overflow_names(mask) for mask in masks]}); "
                + "; ".join(hints)
            )
        if bool(nonfinite.any()):
            reasons.append("non-finite physics state")
        self._fail(bad, f"Tracking simulation failed {stage}: " + "; ".join(reasons))

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        if not bool((torch.isfinite(self.bin_G) & torch.isfinite(self.bin_w_avg)).all()):
            raise RuntimeError("Non-finite motion sampling statistics; inspect the preceding simulation/rewards")
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        next_avg = 0.999 * self.bin_w_avg + 0.001 * soft
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * next_avg / next_avg.sum()
        if not bool((torch.isfinite(probs) & (probs >= 0)).all() & (probs.sum() > 0)):
            raise RuntimeError("Invalid motion sampling probabilities; inspect motion return statistics")
        self.bin_w_avg.copy_(next_avg)
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        p, R, v, w, q, qd, op, oR, ov, ow = (x.clone() for x in m.ref_state(frames))
        quat = mat_to_quat(R)
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.state_joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(q[:, :29], joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(
            self.hand.expand(q[:, 29:41]), joint_ids=self.all_hand_joint_ids, env_ids=env_ids,
        )
        self.object.write_root_link_pose_to_sim(
            torch.cat((op + self.scene.env_origins[env_ids], mat_to_quat(oR)), -1), env_ids=env_ids,
        )
        self.object.write_root_link_velocity_to_sim(torch.cat((ov, ow), -1), env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        self.refresh()

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward()
        self._check_physics("during forward/reset")
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def _episode_progress(self, frames: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """Completion after the next step, relative to the reset-to-target window."""
        clip_end = self.motion.clip_end[self.motion.clip_id[frames]]
        # frames - steps reconstructs the reset frame, including random-phase resets.
        target_steps = (clip_end - frames + steps).clamp(min=1, max=self.max_episode_steps)
        return ((steps + 1).float() / target_steps).clamp(max=1.0)

    def step(self, a: torch.Tensor) -> dict:
        if a.shape != (self.N, ACTION_DIM):
            raise ValueError(f"Expected policy action shape ({self.N}, {ACTION_DIM}), got {tuple(a.shape)}")
        self._ensure_finite(a, "policy action")
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a[:, :BODY_JOINT_DIM]
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        self.robot.set_joint_position_target(
            self.hand.expand(self.hand.action_target(a[:, BODY_JOINT_DIM:])), joint_ids=self.all_hand_joint_ids,
        )
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            torch.maximum(self._peak_nefc, self._nefc, out=self._peak_nefc)
            self.scene.update(self.cfg.physics.mujoco.timestep)
        self._check_physics("after physics substeps")
        S1 = self.state()
        f1 = torch.minimum(f + 1, self.motion.clip_end[self.motion.clip_id[f]])
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        terms.update(self.contact_diagnostics.compute(f1))
        self._ensure_finite(r, "tracking reward")
        term, termination_terms = self.mdp.termination(S1, f1, terms=True)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        termination_terms["clip_end"] = f1 >= clip_end
        termination_terms["time_limit"] = ep_step + 1 >= self.max_episode_steps
        timeout = termination_terms["clip_end"] | termination_terms["time_limit"]
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
            "termination_terms": termination_terms,
            "progress": self._episode_progress(f, ep_step),
            "hand_reference": self.motion.hand_reference(f1),
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * self.mdp.max_step_reward)
            self.reset(ids)
        return out
