"""Per-clip NPZ export, numeric equivalence and offline-loading checks."""

import os

import numpy as np
import pytest
import torch

from supermimic.motion import FRAME_FIELDS, MotionLib, _save_npz, preprocess_directory


class TinyKinematics:
    """Small FK stand-in so NPZ checks need no downloaded assets."""

    def __init__(self):
        self.revision = "model-v1"
        self.forward_calls = 0

    def cache_key(self):
        return {"model_sha256": self.revision, "tracked_bodies": ["root", "tip"]}

    def forward(self, p, R, v, w, q, qd):
        self.forward_calls += 1
        offset = torch.tensor([0.0, 0.0, 0.2], device=p.device)
        return (
            torch.stack((p, p + offset), -2),
            torch.stack((R, R), -3),
            torch.stack((v, v), -2),
            torch.stack((w, w), -2),
        )


@pytest.fixture
def source(tmp_path):
    raw_dir, processed = tmp_path / "raw", tmp_path / "lafan_g1"
    raw_dir.mkdir()
    for i, length in enumerate([64, 95]):
        t = np.arange(length) / 30
        raw = np.zeros((len(t), 36))
        raw[:, 0] = 0.1 * t + i
        raw[:, 2] = 0.8
        raw[:, 5] = np.sin(0.1 * t)
        raw[:, 6] = np.cos(0.1 * t)
        raw[:, 7:] = 0.02 * np.sin(t[:, None] + np.arange(29))
        np.savetxt(raw_dir / f"clip_{i}.csv", raw, delimiter=",")
    return raw_dir, processed, TinyKinematics()


def assert_same(actual, expected):
    assert vars(actual).keys() == vars(expected).keys()
    for name, value in vars(expected).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(actual, name).cpu(), value.cpu(), atol=0, rtol=0)
        elif name != "device":
            assert getattr(actual, name) == value
    frames = torch.tensor([0, 20, actual.num_frames - 1], device=actual.device)
    torch.testing.assert_close(actual.goal(frames, 10).cpu(), expected.goal(frames.to(expected.device), 10).cpu())


def export(source, **kwargs):
    raw, processed, kin = source
    return preprocess_directory(raw, processed, kin, "cpu", **kwargs)


def test_one_npz_per_csv_matches_all_reference_tensors_and_indices(source, monkeypatch):
    raw, processed, kin = source
    expected = MotionLib([str(p) for p in sorted(raw.glob("*.csv"))], kin, "cpu")
    paths = export(source)
    assert [p.name for p in paths] == ["clip_0.npz", "clip_1.npz"]
    for path in paths:
        with np.load(path, allow_pickle=False) as clip:
            assert float(clip["fps"]) == 50
            assert set(FRAME_FIELDS) <= set(clip.files)
            assert clip["q"].shape[1:] == (29,)
            assert all(clip[name].dtype == np.float32 for name in FRAME_FIELDS)

    def unexpected(*args, **kwargs):
        raise AssertionError("NPZ loading must not read CSVs, resample, differentiate or run FK")

    raw.rename(raw.with_name("source_unavailable"))
    monkeypatch.setattr(np, "loadtxt", unexpected)
    monkeypatch.setattr(MotionLib, "__init__", unexpected)
    monkeypatch.setattr(kin, "forward", unexpected)
    assert_same(MotionLib.from_dir(str(processed), kin, "cpu"), expected)


def test_unchanged_npz_files_are_reused(source, monkeypatch):
    _, processed, kin = source
    paths = export(source)
    before = {path.name: path.stat().st_mtime_ns for path in paths}
    calls = kin.forward_calls

    def unexpected(*args, **kwargs):
        raise AssertionError("unchanged NPZs must not be recomputed")

    monkeypatch.setattr(np, "loadtxt", unexpected)
    export(source)
    assert kin.forward_calls == calls
    assert {path.name: path.stat().st_mtime_ns for path in processed.glob("*.npz")} == before


def test_source_content_change_regenerates_only_its_npz(source):
    raw, processed, kin = source
    paths = export(source)
    unchanged = paths[1].stat().st_mtime_ns
    before = MotionLib.from_dir(str(processed), kin, "cpu")
    file = raw / "clip_0.csv"
    stat = file.stat()
    values = np.loadtxt(file, delimiter=",")
    values[:, 7] += 0.1
    np.savetxt(file, values, delimiter=",")
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    export(source)
    assert paths[1].stat().st_mtime_ns == unchanged
    after = MotionLib.from_dir(str(processed), kin, "cpu")
    assert not torch.equal(before.q, after.q)
    assert_same(after, MotionLib([str(p) for p in sorted(raw.glob("*.csv"))], kin, "cpu"))


def test_model_and_source_rate_changes_require_offline_regeneration(source):
    _, processed, kin = source
    export(source)
    before = MotionLib.from_dir(str(processed), kin, "cpu")
    kin.revision = "model-v2"
    with pytest.raises(ValueError, match="model or preprocessing settings changed"):
        MotionLib.from_dir(str(processed), kin, "cpu")
    export(source)
    MotionLib.from_dir(str(processed), kin, "cpu")
    export(source, src_fps=60)
    assert MotionLib.from_dir(str(processed), kin, "cpu").num_frames < before.num_frames


@pytest.mark.parametrize("damage", ["truncated", "missing_array", "wrong_fps", "nonfinite"])
def test_invalid_npz_fails_clearly_and_offline_tool_repairs_it(source, damage):
    _, processed, kin = source
    paths = export(source)
    expected = MotionLib.from_dir(str(processed), kin, "cpu")
    if damage == "truncated":
        paths[0].write_bytes(b"interrupted write")
    else:
        with np.load(paths[0], allow_pickle=False) as clip:
            arrays = dict(clip)
        if damage == "missing_array":
            del arrays["body_R"]
        elif damage == "wrong_fps":
            arrays["fps"] = np.float32(30)
        else:
            arrays["q"][0, 0] = np.nan
        np.savez(paths[0], **arrays)
    with pytest.raises(ValueError, match="Invalid processed motion"):
        MotionLib.from_dir(str(processed), kin, "cpu")
    export(source)
    assert_same(MotionLib.from_dir(str(processed), kin, "cpu"), expected)


def test_force_rebuild(source):
    _, processed, kin = source
    export(source)
    expected = MotionLib.from_dir(str(processed), kin, "cpu")
    calls = kin.forward_calls
    export(source, rebuild=True)
    assert kin.forward_calls > calls
    assert_same(MotionLib.from_dir(str(processed), kin, "cpu"), expected)


def test_interrupted_export_preserves_previous_npz(source, monkeypatch):
    path = export(source)[0]
    before = path.read_bytes()

    def fail_save(stream, **arrays):
        stream.write(b"partial file")
        raise OSError("simulated full disk")

    monkeypatch.setattr(np, "savez", fail_save)
    with pytest.raises(OSError, match="simulated full disk"):
        _save_npz(path, {}, {})
    assert path.read_bytes() == before
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_cpu_npz_loads_on_another_gpu(source):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    _, processed, kin = source
    export(source)
    cpu = MotionLib.from_dir(str(processed), kin, "cpu")
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    gpu = MotionLib.from_dir(str(processed), kin, device)
    assert all(value.device == torch.device(device) for value in vars(gpu).values() if isinstance(value, torch.Tensor))
    assert_same(gpu, cpu)


def test_loader_requires_processed_npz_even_when_csvs_exist(source):
    raw, _, kin = source
    with pytest.raises(FileNotFoundError, match="No processed NPZ"):
        MotionLib.from_dir(str(raw), kin, "cpu")
