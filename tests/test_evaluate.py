"""Joint-angle MAE uses matching reference frames and excludes finished clips."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from supermimic.config import EvalConfig, TrainConfig
from supermimic.evaluate import evaluate_run
from supermimic.mdp import MAX_STEP_REWARD, flatten_state


class SimpleKinematics:
    q_lo = torch.full((29,), -3.0)
    q_hi = torch.full((29,), 3.0)

    def forward(self, p, R, v, w, q, qd):
        return p[:, None], R[:, None], v[:, None], w[:, None]

    def cache_key(self):
        return {"model": "test"}


class EvaluationEnv:
    device = "cpu"
    N = 2

    def __init__(self):
        p = torch.zeros(8, 3)
        p[:, 0] = torch.arange(8) * 0.1
        p[:, 2] = 0.8
        q = torch.arange(8)[:, None].expand(-1, 29) * 0.01
        R = torch.eye(3).repeat(8, 1, 1)
        v, w, qd = torch.zeros(8, 3), torch.zeros(8, 3), torch.zeros(8, 29)
        state = p, R, v, w, q, qd
        self.motion = SimpleNamespace(
            names=["failed", "completed"],
            clip_start=torch.tensor([0, 3]), clip_len=torch.tensor([3, 5]),
            clip_end=torch.tensor([2, 7]), root_p=p, root_R=R, q=q,
            ref_state=lambda frames: tuple(x[frames] for x in state),
            goal=lambda frames, K: torch.zeros(2, 112 * K),
            body_p=p[:, None], body_R=R[:, None],
            hand_q=torch.zeros(8, 24), hand_qd=torch.zeros(8, 24),
            hand_reference=lambda frames: torch.zeros(*frames.shape, 12),
            object_p=p.clone(), object_R=R.clone(), object_v=v, object_w=w,
            object_names=["largebox"] * 2, object_path="test/object.xml", signature="test",
        )
        self.done_eval = torch.zeros(2, dtype=torch.bool)
        self.failed_eval = torch.zeros(2, dtype=torch.bool)
        self.xi = torch.zeros(2, 33)
        self.all_hand_joint_ids = torch.arange(24)
        self.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(2, 24), joint_vel=torch.zeros(2, 24)))

    def reset(self, ids, frames):
        self.f = frames.clone()
        self.t = 0
        self.a_prev = torch.zeros(2, 41)

    def state(self):
        state = [x.clone() for x in self.motion.ref_state(self.f)]
        state[4] += torch.tensor([0.1, 0.2])[:, None] * self.t
        if self.t > 1:
            state[4][0] = float("nan")  # finished worlds must not contaminate aggregates
        state[4] = torch.cat((state[4], self.robot.data.joint_pos), -1)
        state[5] = torch.cat((state[5], self.robot.data.joint_vel), -1)
        return (*state, self.motion.object_p[self.f], self.motion.object_R[self.f],
                self.motion.object_v[self.f], self.motion.object_w[self.f])

    def step(self, action):
        self.t += 1
        self.f = torch.minimum(self.f + 1, torch.tensor([2, 7]))
        self.done_eval[:] = torch.tensor([True, self.t >= 4])
        self.failed_eval[0] = True
        state = self.state()
        diagnostics = {name: torch.zeros(2) for name in (
            "body_position_error", "body_rotation_error",
            "object_position_error", "object_rotation_error", "object_linear_velocity_error",
            "object_angular_velocity_error", "contact_match_rate", "object_penetration",
            "contact_slip_speed", "grasp_loss")}
        diagnostics["joint_mae"] = (state[4][:, :29] - self.motion.q[self.f]).abs().mean(-1)
        return {"r": torch.full((2,), MAX_STEP_REWARD), "s1": flatten_state(state), "terms": diagnostics}


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr("supermimic.evaluate.load_policy", lambda ck, dev: lambda hist, goal: torch.zeros(2, 41))
    env = EvaluationEnv()
    monkeypatch.setattr("torch.cuda.set_device", lambda dev: None)
    monkeypatch.setattr("supermimic.env.make_sim", lambda *args: None)
    monkeypatch.setattr("supermimic.env.TrackingSim", lambda *args, **kwargs: env)
    monkeypatch.setattr("supermimic.kinematics.G1Kinematics", lambda dev: SimpleKinematics())
    monkeypatch.setattr("supermimic.motion.MotionLib.from_dir", lambda *args: env.motion)
    env.mdp = SimpleNamespace(obs=lambda *args: torch.zeros(2, 1), max_step_reward=MAX_STEP_REWARD)
    cfg = OmegaConf.structured(EvalConfig)
    cfg.run_dir = cfg.motion_dir = str(tmp_path)
    cfg.device = "cpu"
    cfg.ckpt = str(tmp_path / "ckpt_000020.pt")
    saved = OmegaConf.to_container(OmegaConf.structured(TrainConfig), resolve=True)
    saved.update(goal_frames=2, hist=3)
    torch.save({"args": saved, "wall_time": 3600, "saved_at": "test"}, cfg.ckpt)
    return env, cfg


def test_evaluate_masks_terminal_samples_and_preserves_recording(evaluation, tmp_path, capsys):
    _, cfg = evaluation
    record = tmp_path / "traj.npz"
    cfg.record = str(record)
    evaluate_run(cfg)
    result = json.loads(capsys.readouterr().out)
    assert result["n_failed"] == 1 and result["failed"] == ["failed"]
    assert result["norm_return"] == pytest.approx(0.75)
    # One failed sample (0.1) plus four completed samples (0.2, 0.4, 0.6, 0.8).
    assert result["mae"] == pytest.approx(0.42)
    assert result["body_position_error"] == result["body_rotation_error"] == 0.0
    with np.load(record) as trajectory:
        np.testing.assert_array_equal(trajectory["alive"], [[True, True], [False, True], [False, True], [False, True]])
        np.testing.assert_array_equal(trajectory["f"], [[0, 3], [1, 4], [1, 5], [1, 6]])
        assert trajectory["q"].shape == (4, 2, 29)
        assert trajectory["hand_q"].shape == trajectory["hand_qd"].shape == (4, 2, 24)
        assert trajectory["hand_reference"].shape == (4, 2, 12)
        assert trajectory["object_p"].shape == (4, 2, 3)
        assert trajectory["ref_hand_q"].shape == (8, 24)
        assert trajectory["ref_object_R"].shape == (8, 3, 3)
        assert trajectory["action"].shape == (4, 2, 41)
    assert not {"hand_mae", "hand_mimic_error"} & result.keys()
    # A second checkpoint resets all accumulation and terminal flags.
    evaluate_run(cfg)
    assert json.loads(capsys.readouterr().out) == result


def test_evaluate_restores_checkpoint_motion_dir_with_explicit_overrides(evaluation, monkeypatch):
    env, cfg = evaluation
    checkpoint = torch.load(cfg.ckpt, weights_only=False)
    checkpoint["args"].update(motion_dir="checkpoint_motions", motion_signature="test")
    torch.save(checkpoint, cfg.ckpt)
    loaded = []

    def load_motion(path, *args):
        loaded.append(path)
        return env.motion

    monkeypatch.setattr("supermimic.motion.MotionLib.from_dir", load_motion)
    cfg.motion_dir = None
    evaluate_run(cfg)
    assert loaded == ["checkpoint_motions"]
    cfg.motion_dir = "override_motions"
    evaluate_run(cfg)
    assert loaded[-1] == "override_motions"


@pytest.mark.parametrize("existing_csv", [False, True])
def test_batch_evaluation_adds_one_mae_column_and_preserves_old_rows(evaluation, tmp_path, existing_csv):
    _, cfg = evaluation
    cfg.ckpt = None
    csv_path = tmp_path / "eval.csv"
    if existing_csv:
        csv_path.write_text(
            "iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n"
            "10,1,test,test,0.5,0.5,0,\n"
        )
    evaluate_run(cfg)
    with csv_path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames[-1] == "mae"
        rows = list(reader)
    assert len(rows) == 1 + existing_csv
    assert rows[-1]["iteration"] == "20"
    assert float(rows[-1]["mae"]) == pytest.approx(0.42)
    if existing_csv:
        assert rows[0]["iteration"] == "10" and rows[0]["mae"] == ""
    result = json.loads((tmp_path / "ckpt_000020_eval.json").read_text())
    assert result["mae"] == pytest.approx(0.42)
    assert result["per_clip"] == {"failed": 0.5, "completed": 1.0}
