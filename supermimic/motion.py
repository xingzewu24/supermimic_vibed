"""Motion dispatch, shared clip indices, and legacy LAFAN preprocessing at 50 Hz."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from .kinematics import G1Kinematics
from .rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50
NPZ_VERSION = 1
FRAME_FIELDS = (
    "root_p",
    "root_R",
    "root_v",
    "root_w",
    "q",
    "qd",
    "body_p",
    "body_R",
    "body_v",
    "body_w",
    "root_v_local",
    "root_w_local",
    "root_gz",
)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _preprocessing_metadata(kin: G1Kinematics) -> dict:
    source_dir = Path(__file__).resolve().parent
    return {
        "format_version": NPZ_VERSION,
        "control_dt": CONTROL_DT,
        "kinematics": kin.cache_key(),
        "preprocessing": {
            name: _file_digest(source_dir / name)
            for name in (
                "motion.py",
                "kinematics.py",
                "rotations.py",
            )
        },
    }


def _read_npz(path: Path, settings: dict, expected_metadata: dict | None = None) -> tuple[dict, dict]:
    """Read numeric arrays without pickle and reject incompatible processed clips."""
    with np.load(path, allow_pickle=False) as archive:
        missing = (set(FRAME_FIELDS) | {"fps", "metadata"}) - set(archive.files)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        metadata = json.loads(str(archive["metadata"].item()))
        if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in settings.items()):
            raise ValueError("model or preprocessing settings changed")
        if expected_metadata is not None and metadata != expected_metadata:
            raise ValueError("source CSV or source frame rate changed")
        if float(archive["fps"].item()) != 1.0 / CONTROL_DT:
            raise ValueError(f"expected {1.0 / CONTROL_DT:g} Hz")
        arrays = {name: archive[name] for name in FRAME_FIELDS}
    frames = arrays["q"].shape[0]
    bodies = len(settings["kinematics"]["tracked_bodies"])
    shapes = {
        "root_p": (frames, 3),
        "root_R": (frames, 3, 3),
        "root_v": (frames, 3),
        "root_w": (frames, 3),
        "q": (frames, 29),
        "qd": (frames, 29),
        "body_p": (frames, bodies, 3),
        "body_R": (frames, bodies, 3, 3),
        "body_v": (frames, bodies, 3),
        "body_w": (frames, bodies, 3),
        "root_v_local": (frames, 3),
        "root_w_local": (frames, 3),
        "root_gz": (frames,),
    }
    if frames < 2:
        raise ValueError("a motion needs at least two frames")
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != shapes[name] or not np.isfinite(array).all():
            raise ValueError(f"invalid {name}: expected finite float32 array with shape {shapes[name]}")
    return arrays, metadata


def _save_npz(path: Path, arrays: dict, metadata: dict) -> None:
    """Atomically save an uncompressed NPZ for fast repeated loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez(
                stream,
                **arrays,
                fps=np.float32(1.0 / CONTROL_DT),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MotionLib:
    def __init__(self, files: list[str], kin: G1Kinematics, device: str, src_fps: float = 30.0):
        if not files:
            raise FileNotFoundError("No motion CSVs found. Run scripts/download_data.sh or set --motion_dir.")
        if src_fps <= 0:
            raise ValueError("source frame rate must be positive")
        self.device = device
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js = [], [], []
        lens = []
        for f in files:
            raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
            if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 36 or not torch.isfinite(raw).all():
                raise ValueError(f"Invalid motion CSV {f}: expected at least two finite rows with 36 columns")
            p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
            quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
            # resample to 50 Hz
            T_src = raw.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.num_frames = int(sum(lens))

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(
                    self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e]
                )
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = (
                torch.cat(bp),
                torch.cat(bR),
                torch.cat(bv),
                torch.cat(bw),
            )
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        self._build_indices(lens)

    def _build_indices(self, lengths: list[int]) -> None:
        """Build cheap global indices after concatenating independently processed clips."""
        lengths = np.asarray(lengths, dtype=np.int64)
        starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1])))
        bin_counts = (lengths + BIN_FRAMES - 1) // BIN_FRAMES
        bin_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(bin_counts[:-1])))
        indices = {
            "clip_len": lengths,
            "clip_start": starts,
            "clip_end": starts + lengths - 1,
            "clip_id": np.repeat(np.arange(len(lengths)), lengths),
            "bin_clip": np.repeat(np.arange(len(lengths)), bin_counts),
            "bin_start": np.concatenate([np.arange(0, n, BIN_FRAMES) + start for n, start in zip(lengths, starts)]),
            "frame_bin": np.concatenate(
                [np.arange(n) // BIN_FRAMES + offset for n, offset in zip(lengths, bin_offsets)]
            ),
        }
        self.num_frames = int(lengths.sum())
        self.num_bins = int(bin_counts.sum())
        for name, array in indices.items():
            setattr(self, name, torch.as_tensor(array, dtype=torch.long, device=self.device))

    @classmethod
    def from_dir(cls, path: str, kin: G1Kinematics, device: str, clips: list[str] | None = None) -> MotionLib:
        """Load split HOI via its adapter, or legacy preprocessed LAFAN clips."""
        files = sorted(Path(path).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No processed NPZ motions in {path}. Run tools/preprocess_motion.py first.")
        if clips is not None:
            unknown = set(clips) - {file.stem for file in files}
            if not clips or unknown or len(set(clips)) != len(clips):
                raise ValueError(f"Invalid motion_clips selection; missing clips: {sorted(unknown)}")
            files = [file for file in files if file.stem in clips]
        try:
            with np.load(files[0], allow_pickle=False) as archive:
                is_hoi = "joint_pos" in archive.files
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f"Invalid processed motion {files[0]}: {error}") from error
        if is_hoi:
            from .hoi import HOIMotionLib

            return HOIMotionLib.load(path, kin, device, clips=clips)
        start = time.perf_counter()
        settings = _preprocessing_metadata(kin)
        states = {name: [] for name in FRAME_FIELDS}
        lengths = []
        for file in files:
            try:
                arrays, _ = _read_npz(file, settings)
            except (OSError, ValueError, KeyError, EOFError, IndexError, zipfile.BadZipFile) as error:
                raise ValueError(
                    f"Invalid processed motion {file}: {error}. Rebuild with tools/preprocess_motion.py --rebuild."
                ) from error
            lengths.append(len(arrays["q"]))
            for name, array in arrays.items():
                states[name].append(torch.from_numpy(array).to(device))
        motion = cls.__new__(cls)
        motion.device = device
        motion.names = [file.stem for file in files]
        for name, values in states.items():
            setattr(motion, name, torch.cat(values))
        motion._build_indices(lengths)
        print(
            f"[motion] Loaded {len(files)} NPZ clips, {motion.num_frames} frames from {path} "
            f"in {time.perf_counter() - start:.2f}s",
            flush=True,
        )
        return motion

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,73)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def preprocess_directory(
    source_dir: str | Path,
    output_dir: str | Path,
    kin: G1Kinematics,
    device: str,
    *,
    src_fps: float = 30.0,
    rebuild: bool = False,
) -> list[Path]:
    """Export each CSV to a same-name NPZ; only changed clips need recomputing."""
    files = sorted(Path(source_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No source CSVs in {source_dir}. Run scripts/download_data.sh or set --source_dir.")
    output_dir = Path(output_dir)
    settings = _preprocessing_metadata(kin)
    pending = []
    for file in files:
        path = output_dir / f"{file.stem}.npz"
        metadata = {**settings, "source_fps": src_fps, "source": {"name": file.name, "sha256": _file_digest(file)}}
        if not rebuild and path.is_file():
            try:
                _read_npz(path, settings, metadata)
                continue
            except (OSError, ValueError, KeyError, EOFError, IndexError, zipfile.BadZipFile):
                pass  # Only the offline tool repairs stale or damaged processed data.
        pending.append((file, path, metadata))
    if pending:
        motion = MotionLib([str(file) for file, _, _ in pending], kin, device, src_fps)
        lengths = motion.clip_len.cpu().tolist()
        start = 0
        for (_, path, metadata), length in zip(pending, lengths):
            arrays = {
                name: getattr(motion, name)[start : start + length].detach().cpu().numpy() for name in FRAME_FIELDS
            }
            _save_npz(path, arrays, metadata)
            print(f"[motion] Wrote {path.name}: {length} frames", flush=True)
            start += length
    print(
        f"[motion] Ready: {len(files)} NPZ files in {output_dir} ({len(pending)} generated, {len(files) - len(pending)} unchanged)",
        flush=True,
    )
    return [output_dir / f"{file.stem}.npz" for file in files]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
