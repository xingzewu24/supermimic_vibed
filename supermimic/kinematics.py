"""Differentiable forward kinematics of the G1 in torch, built from the compiled MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from .robot import ALL_HAND_JOINT_NAMES, G1_JOINT_NAMES, MJCF_PATH, TRACKED_BODIES, URDF_PATH, g1_spec
from .rotations import axis_angle_to_mat, quat_to_mat


class G1Kinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES):
        m = g1_spec().compile()
        self._model_digest = hashlib.sha256(Path(MJCF_PATH).read_bytes()).hexdigest()
        self._urdf_digest = hashlib.sha256(Path(URDF_PATH).read_bytes()).hexdigest()
        self.device = device
        self.joint_names = G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES
        jname_to_idx = {n: i for i, n in enumerate(self.joint_names)}
        # Preserve fixed links, too. MuJoCo orders parents before children.
        bodies, parent, joint_idx, axes, anchors, offs_p, offs_R = [], [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] > 1:
                raise ValueError(f"FK expects at most one joint per body: {name}")
            jid = int(m.body_jntadr[b]) if m.body_jntnum[b] else -1
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if jid >= 0 and m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                parent.append(mj_to_local[int(m.body_parentid[b])])
                if jid >= 0 and m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError(f"FK expects a hinge joint: {name}")
                joint_idx.append(jname_to_idx[m.joint(jid).name] if jid >= 0 else -1)
            axes.append(m.jnt_axis[jid].copy() if jid >= 0 else np.zeros(3))
            anchors.append(m.jnt_pos[jid].copy() if jid >= 0 else np.zeros(3))
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.anchors = torch.tensor(np.array(anchors), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # Palms are fused into the wrist-yaw bodies in the source MJCF.
        self.hand_body_names = [n for n in bodies if n.startswith(("L_", "R_")) or n in (
            "left_wrist_yaw_link", "right_wrist_yaw_link")]
        self._tracked_order = self._ancestors(self.tracked_idx)
        self._hand_idx = [bodies.index(n) for n in self.hand_body_names]
        self._hand_order = self._ancestors(self._hand_idx)
        self._build_hand_surface(m, mj_to_local)
        # Keep the body-only limit API, with full actual limits available separately.
        lo, hi = np.zeros(53), np.zeros(53)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.full_q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.full_q_hi = torch.tensor(hi, dtype=torch.float32, device=device)
        self.q_lo, self.q_hi = self.full_q_lo[:29], self.full_q_hi[:29]

    def cache_key(self) -> dict:
        """Identify the model and the order of the reference's tracked bodies."""
        return {
            "model_sha256": self._model_digest,
            "urdf_sha256": self._urdf_digest,
            "fk_version": 3,
            "joint_names": list(self.joint_names),
            "tracked_bodies": [self.body_names[i] for i in self.tracked_idx],
        }

    def _ancestors(self, indices):
        needed = set(indices)
        for i in indices:
            while self.parent[i] >= 0:
                i = self.parent[i]
                needed.add(i)
        return sorted(needed)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root (pelvis) position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: (...,53), ordered body(29), active fingers(12), followers(12).
                Body-only (...,29) inputs remain supported for body-only outputs.
                Hand outputs require all 24 actual hand coordinates, never mimic reconstruction.
            all_bodies: return every compiled robot body in ``body_names`` order.
        """
        idx = list(range(self.num_bodies)) if all_bodies else self.tracked_idx
        order = list(range(self.num_bodies)) if all_bodies else self._tracked_order
        return self._forward(p, R, v, w, q, qd, idx, order)

    def _forward(self, p, R, v, w, q, qd, idx, order):
        if q.shape != qd.shape or q.shape[-1] not in (29, 53):
            raise ValueError("FK requires matching q/qd with 29 body or 53 full joint coordinates")
        if q.shape[-1] == 29 and any(self.joint_idx[i] >= 29 for i in order):
            raise ValueError("Hand FK requires all 53 joint coordinates, including the 12 actual followers")
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in order:
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            if j >= 0:
                joint_R = axis_angle_to_mat(self.axes[i], q[..., j])
                Ri = R_off @ joint_R
                anchor = (Ri @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                pi = pi + (R_off @ self.anchors[i].unsqueeze(-1)).squeeze(-1) - anchor
                axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                joint_w = axis_w * qd[..., j : j + 1]
                wi = W[par] + joint_w
            else:
                Ri, wi = R_off, W[par]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            if j >= 0:
                vi = vi - torch.cross(joint_w, anchor, dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )

    def forward_hands(self, p, R, v, w, q, qd):
        """Palm and finger link poses/velocities, in ``hand_body_names`` order.

        Palm frames are the wrist-yaw link frames; link origins are not contact points.
        """
        return self._forward(p, R, v, w, q, qd, self._hand_idx, self._hand_order)

    def _build_hand_surface(self, model, mj_to_local):
        """Support vertices of each collision mesh, expressed in its body frame.

        The 26 support directions select vertices on the convex collision hull,
        including the mesh's compiled recentering and the geom's local transform.
        Palm samples come from the hand-base mesh fused into each wrist.
        """
        directions = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1)
                               for z in (-1, 0, 1) if x or y or z])
        points, body_ids, geom_ids = [], [], []
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            b = int(model.geom_bodyid[g])
            body = model.body(b).name
            mesh = int(model.geom_dataid[g])
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or body not in self.hand_body_names:
                continue
            if not body.startswith(("L_", "R_")) and model.mesh(mesh).name not in (
                "L_hand_base_link", "R_hand_base_link"):
                continue
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            verts = model.mesh_vert[start:start + count].astype(np.float64)
            geom_R = np.empty(9)
            mujoco.mju_quat2Mat(geom_R, model.geom_quat[g])
            verts = verts @ geom_R.reshape(3, 3).T + model.geom_pos[g]
            chosen = np.unique((verts @ directions.T).argmax(axis=0))
            points.extend(verts[chosen])
            body_ids.extend([self._hand_idx.index(mj_to_local[b])] * len(chosen))
            geom_ids.extend([g] * len(chosen))
        self.hand_surface_local = torch.tensor(np.array(points), dtype=torch.float32, device=self.device)
        self.hand_surface_body_idx = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        self.hand_surface_geom_ids = geom_ids

    @staticmethod
    def transform_points(poses, body_idx, local_points):
        """World positions and rigid-body velocities at body-local surface points."""
        p, R, v, w = poses
        offset = (R[..., body_idx, :, :] @ local_points.unsqueeze(-1)).squeeze(-1)
        return (p[..., body_idx, :] + offset,
                v[..., body_idx, :] + torch.cross(w[..., body_idx, :], offset, dim=-1))

    def hand_surface(self, p, R, v, w, q, qd):
        """Collision-surface point world positions/velocities from actual full states."""
        return self.transform_points(self.forward_hands(p, R, v, w, q, qd),
                                     self.hand_surface_body_idx, self.hand_surface_local)
