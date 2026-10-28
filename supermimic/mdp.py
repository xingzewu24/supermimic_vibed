"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

State ``S = (p, R, v, w, q, qd, op, oR, ov, ow)``: robot and object link-origin
world poses/velocities. Joints are body(29), active fingers(12), followers(12).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ObjectTrackingConfig, validate_object_config
from .hand import HandControl
from .kinematics import G1Kinematics
from .motion import CONTROL_DT, MotionLib
from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, so3_log, yaw_mat

BODY_JOINT_DIM = 29
HAND_ACTION_DIM = 12
HAND_JOINT_DIM = 24
JOINT_DIM = BODY_JOINT_DIM + HAND_JOINT_DIM
ACTION_DIM = BODY_JOINT_DIM + HAND_ACTION_DIM
ROBOT_STATE_DIM = 13 + 2 * JOINT_DIM
STATE_DIM = ROBOT_STATE_DIM + 13
OBJECT_FEATURE_DIM = 15
OBS_DIM = 99 + 2 * HAND_JOINT_DIM + 2 * HAND_ACTION_DIM + OBJECT_FEATURE_DIM
XI_DIM = 33

# Table A.1 sensor noise (uniform half-widths), plus the reference-relative orientation
OBS_NOISE = torch.cat(
    (
        torch.full((29,), 0.01),
        torch.full((29,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(29),
        torch.full((6,), 0.05),
        torch.full((HAND_JOINT_DIM,), 0.01),
        torch.full((HAND_JOINT_DIM,), 0.5),
        torch.zeros(HAND_ACTION_DIM),
        torch.zeros(OBJECT_FEATURE_DIM),
        torch.zeros(HAND_ACTION_DIM),
    )
)
MAX_STEP_REWARD = 9.0 * CONTROL_DT  # Body terms (5), object position (2) and rotation (2).


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd, op, oR, ov, ow = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd, op, mat_to_quat(oR), ov, ow), -1)


def unflatten_state(x: torch.Tensor):
    if x.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected full state dimension {STATE_DIM}, got {x.shape[-1]}")
    return (x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13],
            x[..., 13:13 + JOINT_DIM], x[..., 13 + JOINT_DIM:ROBOT_STATE_DIM],
            x[..., 119:122], quat_to_mat(x[..., 122:126]), x[..., 126:129], x[..., 129:132])


def body_state(S):
    """Project full simulator/model states onto the unchanged LAFAN body coordinates."""
    return (*S[:4], S[4][..., :BODY_JOINT_DIM], S[5][..., :BODY_JOINT_DIM])


def tracking_errors(q, body_p, body_R, ref_q, ref_body_p, ref_body_R) -> dict[str, torch.Tensor]:
    """Per-transition body tracking errors in rad, m, rad, using world-frame references."""
    return {
        "joint_mae": (q - ref_q).abs().mean(-1),
        "body_position_error": (body_p - ref_body_p).norm(dim=-1).mean(-1),
        "body_rotation_error": so3_log(body_R @ ref_body_R.transpose(-1, -2)).norm(dim=-1).mean(-1),
    }


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S[:6]
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
            object_features(S),
        ),
        -1,
    )


INV_DIM = 1 + 6 + 3 + 3 + 2 * JOINT_DIM + OBJECT_FEATURE_DIM


def object_features(S):
    """Actual object in actual robot heading; world state itself is never re-anchored."""
    p, R = S[:2]
    op, oR, ov, ow = S[6:]
    Ht = yaw_mat(R).transpose(-1, -2)
    local = lambda x: (Ht @ x.unsqueeze(-1)).squeeze(-1)
    return torch.cat((local(op - p), mat_to_6d(Ht @ oR), local(ov), local(ow)), -1)


class MDP:
    def __init__(self, kin: G1Kinematics, motion: MotionLib, default_q: torch.Tensor, cfg: RewardCfg | None = None,
                 hand_cfg=None, object_cfg=None):
        cfg = cfg if cfg is not None else RewardCfg()
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        self.obs_noise = OBS_NOISE.to(dev)
        self.hand = HandControl(dev, hand_cfg)
        self.object = object_cfg or ObjectTrackingConfig()
        validate_object_config(self.object)
        self.max_step_reward = (5.0 + self.object.pos_weight + self.object.rot_weight) * CONTROL_DT

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        _, R, _, w, q, qd = body_state(S)
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1:30]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(Rt @ self.motion.root_R[f])
        return torch.cat((q - self.default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b,
                          a_prev[..., :BODY_JOINT_DIM], rel, S[4][..., BODY_JOINT_DIM:],
                          S[5][..., BODY_JOINT_DIM:], a_prev[..., BODY_JOINT_DIM:],
                          self.motion.hand_reference(f), object_features(S)), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def reward(self, S, f, a, a_prev, terms: bool = False):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f."""
        c, m = self.cfg, self.motion
        p, R, _, _, q, _ = body_state(S)
        bp, bR, bv, bw = self.kin.forward(*S[:6])
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_body_pos = torch.exp(-e_pos / c.body_pos_std**2)
        r_body_ori = torch.exp(-e_ori / c.body_ori_std**2)
        r_body_linvel = torch.exp(-e_lv / c.body_linvel_std**2)
        r_body_angvel = torch.exp(-e_av / c.body_angvel_std**2)
        r_root_pos = 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
        r_root_ori = 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        r_track = r_body_pos + r_body_ori + r_body_linvel + r_body_angvel + r_root_pos + r_root_ori
        r_rate = ((a[..., :BODY_JOINT_DIM] - a_prev[..., :BODY_JOINT_DIM]) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        h = self.hand.cfg
        hand_rate = ((torch.tanh(a[..., BODY_JOINT_DIM:]) - torch.tanh(a_prev[..., BODY_JOINT_DIM:])) ** 2).mean(-1)
        all_hand_q = S[4][..., BODY_JOINT_DIM:]
        hand_limit = (torch.relu(all_hand_q - self.hand.full_hi) + torch.relu(self.hand.full_lo - all_hand_q)).sum(-1)
        op, oR, ov, ow = S[6:]
        object_pos_sq = (op - m.object_p[f]).square().sum(-1)
        # Shortest SO(3) angle, equivalent to the reference's quat_error_magnitude.
        object_rot_sq = so3_log(m.object_R[f].transpose(-1, -2) @ oR).square().sum(-1)
        r_object_pos = self.object.pos_weight * torch.exp(-object_pos_sq / self.object.pos_std**2)
        r_object_rot = self.object.rot_weight * torch.exp(-object_rot_sq / self.object.rot_std**2)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim
             + r_object_pos + r_object_rot
             - h.action_rate_weight * hand_rate - h.joint_limit_weight * hand_limit) * CONTROL_DT
        if terms:
            return r, {
                **tracking_errors(q, bp, bR, m.q[f], m.body_p[f], m.body_R[f]),
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
                "object_position_error": object_pos_sq.sqrt(),
                "object_rotation_error": object_rot_sq.sqrt(),
                "object_linear_velocity_error": (ov - m.object_v[f]).norm(dim=-1),
                "object_angular_velocity_error": (ow - m.object_w[f]).norm(dim=-1),
                "reward_object_pos": r_object_pos * CONTROL_DT,
                "reward_object_rot": r_object_rot * CONTROL_DT,
                "reward_body_pos": r_body_pos * CONTROL_DT,
                "reward_body_ori": r_body_ori * CONTROL_DT,
                "reward_body_linvel": r_body_linvel * CONTROL_DT,
                "reward_body_angvel": r_body_angvel * CONTROL_DT,
                "reward_root_pos": r_root_pos * CONTROL_DT,
                "reward_root_ori": r_root_ori * CONTROL_DT,
                "reward_action_rate": c.w_action_rate * r_rate * CONTROL_DT,
                "reward_joint_limit": c.w_joint_limit * r_lim * CONTROL_DT,
                "reward_hand_action_rate": -h.action_rate_weight * hand_rate * CONTROL_DT,
                "reward_hand_joint_limit": -h.joint_limit_weight * hand_limit * CONTROL_DT,
            }
        return r

    def termination(self, S, f, terms: bool = False):
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        height = e_z > self.cfg.term_height
        tilt = e_tilt > self.cfg.term_tilt
        terminated = height | tilt
        if terms:
            return terminated, {"height": height, "tilt": tilt, "height_error": e_z, "tilt_error": e_tilt}
        return terminated

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi) -> torch.Tensor:
        """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
        m = self.motion
        p, R, _, _, q, _ = body_state(S)
        H = yaw_mat(R)
        Ht = H.transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(*S[:6])[0]
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        op, oR, ov, ow = S[6:]
        local = lambda x: (Ht @ x.unsqueeze(-1)).squeeze(-1)
        object_error = torch.cat((local(m.object_p[f] - op), so3_log(oR.transpose(-1, -2) @ m.object_R[f]),
                                  local(m.object_v[f] - ov), local(m.object_w[f] - ow)), -1)
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi,
                          m.hand_reference(f), object_error), -1)


CRITIC_EXTRA = 29 + 14 * 3 + 3 + 6 + XI_DIM + HAND_ACTION_DIM + 12


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:30] / 0.01, xi[..., 30:33] / 0.05), -1)
