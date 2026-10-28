"""Read-only adapter for split 50 Hz HOI references, using link-origin velocities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .hand import HandControl
from .motion import CONTROL_DT, MotionLib, _file_digest
from .robot import ALL_HAND_JOINT_NAMES, G1_JOINT_NAMES, MJCF_PATH, URDF_PATH, g1_spec, object_spec
from .rotations import mat_to_6d, quat_to_mat, so3_log, yaw_mat

GOAL_FRAME_DIM = 112  # body 73 + active hand q/qd 24 + object pose/velocity 15
ADAPTER_VERSION = 2


def com_to_link_velocity(v_com, omega_world, rotation, com_offset):
    return v_com - torch.cross(omega_world, (rotation @ com_offset.unsqueeze(-1)).squeeze(-1), dim=-1)


def _validate_raw(raw):
    T = len(raw["joint_pos"])
    if T < 2 or raw["motion_lengths"].shape != (1,) or int(raw["motion_lengths"][0]) != T:
        raise ValueError("expected one pre-split clip with at least two frames")
    if raw["fps"].size != 1 or float(raw["fps"].item()) != 1 / CONTROL_DT:
        raise ValueError("expected 50 Hz; resampling is not implicit")
    if raw["motion_names"].shape != (1,) or raw["object_names"].shape != (1,):
        raise ValueError("expected one motion name and object name")
    if raw["joint_names"].ndim != 1 or raw["body_names"].ndim != 1:
        raise ValueError("joint_names and body_names must be one-dimensional name arrays")
    joints, bodies = raw["joint_names"].tolist(), raw["body_names"].tolist()
    if len(joints) != 53 or set(joints) != set(G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES):
        raise ValueError("joint_names must contain exactly the 53 robot joints")
    if len(bodies) != 54 or len(set(bodies)) != 54:
        raise ValueError("invalid body_names")
    shapes = {"joint_pos": (T, 53), "joint_vel": (T, 53), "body_pos_w": (T, 54, 3),
              "body_quat_w": (T, 54, 4), "body_lin_vel_w": (T, 54, 3), "body_ang_vel_w": (T, 54, 3),
              "object_pos_w": (T, 3), "object_quat_w": (T, 4), "object_lin_vel_w": (T, 3),
              "object_ang_vel_w": (T, 3), "contact_label": (T, 54)}
    for key, shape in shapes.items():
        if raw[key].shape != shape or not np.isfinite(raw[key]).all():
            raise ValueError(f"invalid {key}; expected finite {shape}")
    if not np.isin(raw["contact_label"], [-1, 0, 1]).all():
        raise ValueError("invalid contact labels")
    for key in ("body_quat_w", "object_quat_w"):
        if np.max(np.abs(np.linalg.norm(raw[key], axis=-1) - 1)) > 1e-3:
            raise ValueError(f"non-unit wxyz quaternion: {key}")
    if str(raw["object_names"][0]) != "largebox":
        raise ValueError("first HOI task supports only the fixed largebox asset")
    return T, joints, bodies


class HOIMotionLib(MotionLib):
    goal_frame_dim = GOAL_FRAME_DIM
    is_hoi = True

    @classmethod
    def load(cls, directory: str, kin, device: str, clips=None):
        # Asset validation must be independent of the training matmul precision.
        # TF32 on chains of small rotation matrices can amplify FK errors to millimetres.
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            return cls._load(directory, kin, device, clips)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32

    @classmethod
    def _load(cls, directory: str, kin, device: str, clips):
        path = Path(directory).resolve()
        root = next((p for p in (path, *path.parents) if (p / "manifest.json").is_file()), None)
        if root is None:
            raise FileNotFoundError(f"No HOI manifest.json above {path}")
        manifest = json.loads((root / "manifest.json").read_text())
        expected = {"quaternion": "wxyz", "up_axis": "z", "length_unit": "meter",
                    "linear_velocity": "world_center_of_mass", "position": "world_link_origin",
                    "angular_velocity": "world", "control_dt": CONTROL_DT}
        if any(manifest["conventions"].get(k) != v for k, v in expected.items()):
            raise ValueError("Unsupported HOI coordinate/velocity conventions")
        for key, asset in (("mjcf_sha256", MJCF_PATH), ("urdf_sha256", URDF_PATH)):
            if manifest["robot"][key] != _file_digest(Path(asset)):
                raise ValueError(f"HOI robot asset mismatch: {asset}")
        for name, digest in manifest["robot"]["mesh_sha256"].items():
            if _file_digest(Path(MJCF_PATH).parent / name) != digest:
                raise ValueError(f"HOI robot mesh mismatch: {name}")
        model, hand = g1_spec().compile(), HandControl(device)
        object_path = root / "objects/largebox/object.xml"
        for key, asset in (("mjcf_sha256", object_path), ("mesh_sha256", object_path.parent / "largebox.obj")):
            if _file_digest(asset) != manifest["objects"]["largebox"][key]:
                raise ValueError(f"HOI object asset mismatch: {asset}")
        om = object_spec(str(object_path)).compile()
        object_com = torch.tensor(om.body("largebox").ipos, device=device, dtype=torch.float32)
        out = cls.__new__(cls)
        out.device, out.names, out.object_names = device, [], []
        out.validation = {"adapter_version": ADAPTER_VERSION, "clips": []}
        fields = {k: [] for k in ("root_p", "root_R", "root_v", "root_w", "full_q", "full_qd",
                                  "body_p", "body_R", "body_v", "body_w", "object_p", "object_R",
                                  "object_v", "object_w", "contact_label")}
        lengths, signatures = [], []
        tracked = kin.cache_key()["tracked_bodies"]
        for file in sorted(path.glob("*.npz")):
            if clips is not None and file.stem not in clips:
                continue
            try:
                with np.load(file, allow_pickle=False) as archive:
                    raw = {k: archive[k] for k in archive.files}
                name_sources = {}
                for key in ("joint_names", "body_names"):
                    if key in raw:
                        name_sources[key] = "motion"
                    elif key in manifest:
                        raw[key] = np.asarray(manifest[key])
                        name_sources[key] = "manifest"
                    else:
                        raise ValueError(f"missing {key} in both the motion NPZ and manifest.json")
                T, joints, bodies = _validate_raw(raw)
                ix = [joints.index(n) for n in G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES]
                bi, ri = [bodies.index(n) for n in tracked], bodies.index("pelvis")
                def tensor(k, arrays=raw):
                    return torch.as_tensor(arrays[k], device=device, dtype=torch.float32)
                bp, bR, bw = tensor("body_pos_w"), quat_to_mat(tensor("body_quat_w")), tensor("body_ang_vel_w")
                offsets = torch.tensor(np.array([model.body(n).ipos for n in bodies]), device=device, dtype=torch.float32)
                bv = com_to_link_velocity(tensor("body_lin_vel_w"), bw, bR, offsets)
                full_q, full_qd = tensor("joint_pos")[:, ix], tensor("joint_vel")[:, ix]
                hq, hqd = full_q[:, 29:], full_qd[:, 29:]
                mimic_error = (hq - hand.expand(hq[:, :12])).abs().max().item()
                mimic_vel_error = (hqd[:, 12:] - hqd[:, hand.drivers] * hand.ratios).abs().max().item()
                if mimic_error > 1e-4 or mimic_vel_error > 1e-3:
                    raise ValueError("reference violates hand mimic relationships")
                op, oR, ow = tensor("object_pos_w"), quat_to_mat(tensor("object_quat_w")), tensor("object_ang_vel_w")
                ov = com_to_link_velocity(tensor("object_lin_vel_w"), ow, oR, object_com)
                values = (bp[:, ri], bR[:, ri], bv[:, ri], bw[:, ri], full_q, full_qd,
                          bp[:, bi], bR[:, bi], bv[:, bi], bw[:, bi], op, oR, ov, ow, tensor("contact_label"))
                report = _consistency_report(file, values, kin, hand, mimic_error, mimic_vel_error,
                                             bodies, (bp, bR, bv, bw))
                report["name_sources"] = name_sources
                out.validation["clips"].append(report)
                for key, value in zip(fields, values, strict=True):
                    fields[key].append(value)
                out.names.append(str(raw["motion_names"][0]))
                out.object_names.append("largebox")
                lengths.append(T)
                signatures.append((file.name, _file_digest(file)))
                if hasattr(out, "body_names") and bodies != out.body_names:
                    raise ValueError("inconsistent contact-label body order across clips")
                out.body_names = bodies
            except (KeyError, ValueError) as error:
                raise ValueError(f"Invalid HOI motion {file}: {error}") from error
        if not lengths:
            raise FileNotFoundError(f"No HOI NPZ files in {path}")
        if len(set(out.names)) != len(out.names):
            raise ValueError("Duplicate HOI motion names")
        for key, parts in fields.items():
            setattr(out, key, torch.cat(parts))
        out.q, out.qd = out.full_q[:, :29], out.full_qd[:, :29]
        out.hand_q, out.hand_qd = out.full_q[:, 29:], out.full_qd[:, 29:]
        out.object_path = str(object_path)
        Rt = out.root_R.transpose(-1, -2)
        out.root_v_local = (Rt @ out.root_v.unsqueeze(-1)).squeeze(-1)
        out.root_w_local = (Rt @ out.root_w.unsqueeze(-1)).squeeze(-1)
        out.root_gz = out.root_R[:, 2, 2]
        out._build_indices(lengths)
        signature = {"version": ADAPTER_VERSION, "clips": signatures, "robot": kin.cache_key(),
                     "manifest": _file_digest(root / "manifest.json"),
                     "object": [(p.name, _file_digest(p)) for p in sorted(object_path.parent.glob("*"))
                                if p.suffix in (".xml", ".obj")]}
        out.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        violations = sum(len(r["frames"]) for c in out.validation["clips"] for r in c["limit_violations"])
        print(f"[hoi] Loaded {len(lengths)} clips, {out.num_frames} frames; "
              f"{violations} frame/joint limit violations reported; references preserved.", flush=True)
        return out

    def hand_reference(self, f):
        return self.hand_q[f][..., :12]

    def goal(self, f, K):
        fut = self.future_frames(f, K)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(-3)
        origin = self.root_p[f].unsqueeze(-2)
        transform = lambda x: (Ht @ x.unsqueeze(-1)).squeeze(-1)
        return torch.cat((transform(self.root_p[fut] - origin), mat_to_6d(Ht @ self.root_R[fut]),
                          self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1,
                          self.hand_q[fut][..., :12], self.hand_qd[fut][..., :12] * 0.1,
                          transform(self.object_p[fut] - origin), mat_to_6d(Ht @ self.object_R[fut]),
                          transform(self.object_v[fut]), transform(self.object_w[fut])), -1).flatten(-2)

    def ref_state(self, f):
        return (self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.full_q[f], self.full_qd[f],
                self.object_p[f], self.object_R[f], self.object_v[f], self.object_w[f])


def _consistency_report(file, values, kin, hand, mimic_error, mimic_vel_error, body_names, full_body):
    p, R, v, w, q, qd, bp, bR, bv, bw = values[:10]
    full_fk = kin.forward(p, R, v, w, q, qd, all_bodies=True)
    index = [body_names.index(n) for n in kin.body_names]

    def fk_errors(actual, reference):
        rp, rR, rv, rw = reference
        return {"position_m": (actual[0] - rp).abs().max().item(),
                "rotation_rad": so3_log(actual[1] @ rR.transpose(-1, -2)).norm(dim=-1).max().item(),
                "linear_velocity_m_s": (actual[2] - rv).abs().max().item(),
                "angular_velocity_rad_s": (actual[3] - rw).abs().max().item()}

    full_errors = fk_errors(full_fk, tuple(x[:, index] for x in full_body))
    errors = fk_errors(tuple(x[:, kin.tracked_idx] for x in full_fk), (bp, bR, bv, bw))
    if max(full_errors.values()) > 1e-3:
        raise ValueError(f"reference FK mismatch (including hands): {full_errors}")
    lo, hi = torch.cat((kin.q_lo, hand.lo)), torch.cat((kin.q_hi, hand.hi))
    excess = torch.relu(q[:, :41] - hi) + torch.relu(lo - q[:, :41])
    records = []
    for j, name in enumerate(G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES[:12]):
        frames = (excess[:, j] > 1e-5).nonzero().flatten()
        if len(frames):
            records.append({"joint": name, "frames": frames.cpu().tolist(),
                            "max_excess_rad": excess[:, j].max().item(),
                            "required_contact_frames": frames[(values[-1][frames] == 1).any(-1)].cpu().tolist()})
    return {"file": file.name, "frames": len(q), "fk_max_error": errors, "full_fk_max_error": full_errors,
            "mimic_max_rad": mimic_error,
            "mimic_velocity_max": mimic_vel_error, "limit_violations": records}
