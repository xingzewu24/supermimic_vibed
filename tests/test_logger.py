"""Exercise the real offline SDK without contacting SwanLab or allocating a GPU."""

import csv
import math
import os
import sys

import pytest
from omegaconf import OmegaConf

from supermimic import logger as logging_module
from supermimic.config import TrainConfig
from supermimic.logger import (
    PERFORMANCE_METRICS,
    RETIRED_TRAIN_METRICS,
    REWARD_METRICS,
    TRACKING_ERROR_METRICS,
    TRACKING_METRICS,
    TRAIN_METRICS,
    TrackingMetrics,
    create_training_logger,
    group_training_metrics,
    prepare_training_log,
)


@pytest.mark.parametrize("existing_key", [None, "existing-environment-key"])
def test_offline_logger_loads_credentials_and_records_metrics(tmp_path, monkeypatch, existing_key):
    monkeypatch.setattr(logging_module, "REPO", tmp_path)
    monkeypatch.setenv("SWANLAB_ROOT", str(tmp_path / "swanlab_root"))
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    if existing_key:
        monkeypatch.setenv("SWANLAB_API_KEY", existing_key)
    local_key = "local-test-credentials"
    (tmp_path / ".env").write_text(f"SWANLAB_API_KEY={local_key}\n")
    cfg = OmegaConf.structured(TrainConfig)
    cfg.logger.mode = "offline"
    cfg.log_dir = str(tmp_path / "training")
    config_data = OmegaConf.to_container(cfg, resolve=True)

    with create_training_logger(cfg, config_data) as run:
        assert run.mode == "offline"
        assert run.config["run_name"] == cfg.run_name
        assert "SWANLAB_API_KEY" not in run.config
        assert os.environ["SWANLAB_API_KEY"] == (existing_key or local_key)
        metrics = {name: float(index) for index, name in enumerate(TRAIN_METRICS)}
        metrics.update(iteration=50, sim_steps=1000)
        metrics.update({name: 1.0 for name in RETIRED_TRAIN_METRICS})
        grouped = group_training_metrics(metrics)
        assert len(grouped) == len(TRAIN_METRICS) - 1
        assert {key.split("/", 1)[0] for key in grouped} == {
            "performance", "reward", "failure", "world_model", "policy", "train",
        }
        assert grouped["performance/episode_return"] == metrics["ep_ret_norm"]
        assert grouped["performance/success_rate"] == metrics["success_rate"]
        assert grouped["performance/progress"] == metrics["progress"]
        for name in TRACKING_ERROR_METRICS:
            assert grouped[f"performance/{name}"] == metrics[name]
        assert grouped["reward/total"] == metrics["sim_rew"]
        assert grouped["reward/action_rate"] == metrics["reward_action_rate"]
        assert grouped["failure/failure_fraction"] == metrics["term_frac"]
        assert {key for key in grouped if key.startswith("failure/")} == {
            "failure/failure_fraction", "failure/height_fraction", "failure/tilt_fraction",
        }
        assert grouped["world_model/wm_loss"] == metrics["wm_loss"]
        assert grouped["world_model/wm_err8"] == metrics["wm_err8"]
        assert grouped["world_model/wm_body_err8"] == metrics["wm_body_err8"]
        assert not {"world_model/wm_hand_err8", "performance/hand_mae", "performance/hand_mimic_error"} & grouped.keys()
        assert grouped["policy/loss"] == metrics["pi_loss"]
        assert grouped["train/critic_loss"] == metrics["v_loss"]
        assert grouped["train/learning_rate"] == metrics["lr"]
        assert "train/iteration" not in grouped
        run.log(grouped, step=50)
        assert run.alive

    assert not run.alive
    log_dir = tmp_path / "training" / "swanlab"
    records = list(log_dir.rglob("*.swanlab"))
    assert len(records) == 1 and records[0].stat().st_size > 0
    for path in log_dir.rglob("*"):
        if path.is_file():
            assert (existing_key or local_key).encode() not in path.read_bytes()
    assert local_key not in repr(config_data)


def test_pretraining_skips_unobserved_metrics_and_keeps_real_zeros():
    metrics = {
        "iteration": 50,
        "wm_loss": 0.0,
        "sim_rew": 0.0,
        "lr": 0.0002,
        "pi_loss": float("nan"),
        "v_loss": None,
        "imag_rew": float("inf"),
        "imag_survival": float("-inf"),
        "ep_ret_norm": float("nan"),
        "term_frac": float("nan"),
    }
    assert group_training_metrics(metrics) == {
        "reward/total": 0.0,
        "world_model/wm_loss": 0.0,
        "train/learning_rate": 0.0002,
    }
    assert group_training_metrics({}) == {}


def test_policy_metrics_start_when_observed_without_a_critic():
    assert group_training_metrics({
        "pi_loss": 0.0,
        "pi_gradnorm": 0.0,
        "imag_rew": 0.7,
        "imag_survival": 1.0,
        "v_loss": float("nan"),
        "term_frac": 0.0,
    }) == {
        "failure/failure_fraction": 0.0,
        "policy/loss": 0.0,
        "policy/grad_norm": 0.0,
        "policy/imagined_reward": 0.7,
        "policy/imagined_survival": 1.0,
    }


def diagnostic_transition(height, tilt, clip_end, time_limit, height_error, tilt_error, scale=1.0, progress=None):
    import torch

    height, tilt, clip_end, time_limit = (
        torch.tensor(values, dtype=torch.bool) for values in (height, tilt, clip_end, time_limit)
    )
    terms = {
        name: torch.full((height.numel(),), scale * (index + 1))
        for index, name in enumerate(REWARD_METRICS)
    }
    # A fixed negative example keeps aggregation checks independent of reward column order.
    terms["reward_action_rate"] = torch.full((height.numel(),), -8 * scale)
    terms.update({
        name: torch.full((height.numel(),), scale * (index + 1) * 0.1)
        for index, name in enumerate(TRACKING_ERROR_METRICS)
    })
    term = height | tilt
    return {
        "term": term,
        "done": term | clip_end | time_limit,
        "progress": torch.ones(height.numel()) if progress is None else torch.tensor(progress),
        "terms": terms,
        "termination_terms": {
            "height": height, "tilt": tilt, "clip_end": clip_end, "time_limit": time_limit,
            "height_error": torch.tensor(height_error), "tilt_error": torch.tensor(tilt_error),
        },
    }


def test_tracking_metrics_count_overlapping_failures_and_weight_samples():
    metrics = TrackingMetrics("cpu", max_step_reward=2.0)
    metrics.update(diagnostic_transition(
        height=[True, False, True, False, False, False],
        tilt=[False, True, True, False, False, False],
        clip_end=[True, False, False, True, True, False],
        time_limit=[False, False, False, True, False, True],
        height_error=[0.3, 0.0, 0.4, 0.0, 0.0, 0.0],
        tilt_error=[0.0, 0.9, 1.0, 0.0, 0.0, 0.0],
        progress=[0.25, 0.5, 0.75, 1.0, 1.0, 1.0],
    ))
    metrics.update(diagnostic_transition(
        height=[False, False], tilt=[False, False], clip_end=[False, False], time_limit=[False, False],
        height_error=[0.1, 0.0], tilt_error=[0.1, 0.0], scale=3.0,
        progress=[0.2, 0.8],
    ))
    result = metrics.summary()
    assert result["term_frac"] == pytest.approx(3 / 6)
    assert result["success_rate"] == pytest.approx(3 / 6)
    assert result["progress"] == pytest.approx(4.5 / 6)
    assert result["joint_mae"] == pytest.approx(0.15)
    assert result["body_position_error"] == pytest.approx(0.30)
    assert result["body_rotation_error"] == pytest.approx(0.45)
    assert result["term_height_frac"] == pytest.approx(2 / 6)
    assert result["term_tilt_frac"] == pytest.approx(2 / 6)
    assert result["term_frac"] + result["success_rate"] == pytest.approx(1)
    assert not set(RETIRED_TRAIN_METRICS) & result.keys()
    assert result["reward_body_pos"] == pytest.approx((6 + 2 * 3) / 8 / 2)
    assert result["reward_action_rate"] == pytest.approx(-8 * (6 + 2 * 3) / 8 / 2)
    assert all(isinstance(value, float) for value in result.values())
    metrics.reset()
    assert metrics.summary() == {}


def test_tracking_metrics_omit_episode_fractions_until_an_episode_ends():
    metrics = TrackingMetrics("cpu", max_step_reward=0.12)
    metrics.update(diagnostic_transition(
        height=[False], tilt=[False], clip_end=[False], time_limit=[False],
        height_error=[0.0], tilt_error=[0.0], scale=0.0,
    ))
    result = metrics.summary()
    assert math.isnan(result["term_frac"])
    assert math.isnan(result["term_height_frac"])
    assert math.isnan(result["success_rate"])
    assert math.isnan(result["progress"])
    grouped = group_training_metrics(result)
    assert "failure/failure_fraction" not in grouped
    assert "failure/height_fraction" not in grouped
    assert "performance/success_rate" not in grouped
    assert "performance/progress" not in grouped
    assert not any(key.startswith("failure/") for key in grouped)
    assert grouped["reward/body_position"] == 0.0
    for name in TRACKING_ERROR_METRICS:
        assert grouped[f"performance/{name}"] == 0.0


@pytest.mark.parametrize("added_columns", [TRACKING_METRICS, tuple(PERFORMANCE_METRICS), tuple(TRACKING_ERROR_METRICS)])
def test_training_csv_extends_old_header_without_changing_history(tmp_path, added_columns):
    path = tmp_path / "train_log.csv"
    old_columns = tuple(name for name in TRAIN_METRICS if name not in added_columns)
    old_row = {"iteration": "50", "sim_steps": "10000000001", "wm_loss": "0", "pi_loss": ""}
    if "reward_body_pos" in old_columns:
        old_row["reward_body_pos"] = "0.125"
    if "success_rate" in old_columns:
        old_row.update(success_rate="0.5", progress="0.75")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow(old_row)
    prepare_training_log(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == list(TRAIN_METRICS)
    assert len(rows) == 1
    assert {name: rows[0][name] for name in old_row} == old_row
    assert all(rows[0][name] == "" for name in added_columns)
    content = path.read_bytes()
    prepare_training_log(path)
    assert path.read_bytes() == content
    assert not list(tmp_path.glob(".train_log.csv.*"))


def test_training_csv_initializes_header_and_preserves_unknown_schema(tmp_path):
    path = tmp_path / "train_log.csv"
    assert prepare_training_log(path) == TRAIN_METRICS
    with path.open(newline="") as stream:
        assert next(csv.reader(stream)) == list(TRAIN_METRICS)
    path.write_text("iteration,unknown_metric\n50,123\n")
    content = path.read_bytes()
    with pytest.raises(ValueError, match="unknown CSV schema"):
        prepare_training_log(path)
    assert path.read_bytes() == content


def test_training_csv_preserves_retired_history_without_adding_retired_columns_to_new_logs(tmp_path):
    path = tmp_path / "train_log.csv"
    old_columns = ("iteration", "term_frac", *RETIRED_TRAIN_METRICS)
    old_row = {name: str(index / 10) for index, name in enumerate(old_columns)}
    old_row["iteration"] = "50"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow(old_row)
    columns = prepare_training_log(path)
    assert columns == TRAIN_METRICS + RETIRED_TRAIN_METRICS
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == list(columns)
    assert len(rows) == 1
    assert {name: rows[0][name] for name in old_columns} == old_row
    assert rows[0]["success_rate"] == ""
    content = path.read_bytes()
    assert prepare_training_log(path) == columns
    assert path.read_bytes() == content
    new_path = tmp_path / "new.csv"
    assert prepare_training_log(new_path) == TRAIN_METRICS
    with new_path.open(newline="") as stream:
        assert not set(next(csv.reader(stream))) & set(RETIRED_TRAIN_METRICS)


@pytest.mark.parametrize("failed,clip_end,time_limit,success", [
    (False, True, False, 1.0),
    (False, False, True, 1.0),
    (True, True, True, 0.0),
])
def test_performance_success_requires_no_failure_even_at_full_progress(failed, clip_end, time_limit, success):
    metrics = TrackingMetrics("cpu", max_step_reward=0.12)
    metrics.update(diagnostic_transition(
        height=[failed], tilt=[False], clip_end=[clip_end], time_limit=[time_limit],
        height_error=[0.3 if failed else 0.0], tilt_error=[0.0], progress=[1.0],
    ))
    grouped = group_training_metrics(metrics.summary())
    assert grouped["performance/success_rate"] == success
    assert grouped["performance/progress"] == 1.0


@pytest.mark.parametrize("backend, mode", [("none", "online"), ("swanlab", "disabled")])
def test_disabled_logger_needs_no_sdk_or_credentials(tmp_path, monkeypatch, backend, mode):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    monkeypatch.setitem(sys.modules, "dotenv", None)
    cfg = OmegaConf.structured(TrainConfig)
    cfg.logger.backend = backend
    cfg.logger.mode = mode
    cfg.log_dir = str(tmp_path)

    with create_training_logger(cfg, OmegaConf.to_container(cfg, resolve=True)) as run:
        assert run is None

    assert not list(tmp_path.iterdir())
