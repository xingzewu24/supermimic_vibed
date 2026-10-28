"""Experiment tracking with credentials kept outside saved training configs."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

from supermimic.config import REPO

REWARD_METRICS = {
    "reward_body_pos": "body_position",
    "reward_body_ori": "body_orientation",
    "reward_body_linvel": "body_linear_velocity",
    "reward_body_angvel": "body_angular_velocity",
    "reward_root_pos": "root_position",
    "reward_root_ori": "root_orientation",
    "reward_action_rate": "action_rate",
    "reward_joint_limit": "joint_limit",
    "reward_hand_action_rate": "hand_action_rate",
    "reward_hand_joint_limit": "hand_joint_limit",
    "reward_object_pos": "object_position",
    "reward_object_rot": "object_orientation",
}
TERMINATION_METRICS = {
    "term_height_frac": "height_fraction",
    "term_tilt_frac": "tilt_fraction",
}
RETIRED_TRAIN_METRICS = (
    "term_clip_end_frac", "term_time_limit_frac", "term_height_error", "term_tilt_error",
    "term_height_error_max", "term_tilt_error_max", "reward_hand_velocity", "reward_hand_track",
    "hand_mae", "hand_mimic_error", "wm_hand_err8",
)
TRACKING_ERROR_METRICS = {
    "joint_mae": "joint_mae",
    "body_position_error": "body_position_error",
    "body_rotation_error": "body_rotation_error",
    "object_position_error": "object_position_error",
    "object_rotation_error": "object_rotation_error",
    "object_linear_velocity_error": "object_linear_velocity_error",
    "object_angular_velocity_error": "object_angular_velocity_error",
    "contact_match_rate": "contact_match_rate",
    "object_penetration": "object_penetration",
    "contact_slip_speed": "contact_slip_speed",
    "grasp_loss": "grasp_loss",
}
PERFORMANCE_METRICS = {"success_rate": "success_rate", "progress": "progress", **TRACKING_ERROR_METRICS}
TRACKING_METRICS = tuple(REWARD_METRICS) + tuple(TERMINATION_METRICS) + tuple(PERFORMANCE_METRICS)
TRAIN_METRICS = (
    "iteration", "wall_h", "sim_steps", "wm_loss", "wm_err8", "pi_loss", "v_loss", "imag_rew",
    "pi_gradnorm", "imag_survival", "sim_rew", "ep_len", "ep_ret_norm", "term_frac", "it_per_s", "lr",
    "wm_body_err8",
) + TRACKING_METRICS + ("wm_object_err8", "wm_object_contact_err8")

SWANLAB_METRIC_GROUPS = {
    "performance": {
        **PERFORMANCE_METRICS,
        "ep_ret_norm": "episode_return",
        "ep_len": "episode_length",
    },
    "reward": {"sim_rew": "total", **REWARD_METRICS},
    "failure": {"term_frac": "failure_fraction", **TERMINATION_METRICS},
    "world_model": {
        "wm_loss": "wm_loss",
        "wm_err8": "wm_err8",
        "wm_body_err8": "wm_body_err8",
        "wm_object_err8": "wm_object_err8",
        "wm_object_contact_err8": "wm_object_contact_err8",
    },
    "policy": {
        "pi_loss": "loss",
        "pi_gradnorm": "grad_norm",
        "imag_rew": "imagined_reward",
        "imag_survival": "imagined_survival",
    },
    "train": {
        "v_loss": "critic_loss",
        "wall_h": "wall_hours",
        "sim_steps": "sim_steps",
        "it_per_s": "iterations_per_second",
        "lr": "learning_rate",
    },
}


class TrackingMetrics:
    """Accumulate transition diagnostics on device; transfer scalars only when logging."""

    def __init__(self, device, max_step_reward: float):
        import torch

        self.max_step_reward = max_step_reward
        # Reward/tracking-error sums, failure counts, successes, and completed-episode progress.
        self.sums = torch.zeros(len(REWARD_METRICS) + len(TRACKING_ERROR_METRICS) + 6, device=device)
        self.samples = 0

    def update(self, transition: dict) -> None:
        import torch

        term, done = transition["term"], transition["done"]
        diagnostics = transition["termination_terms"]
        columns = [transition["terms"][name] for name in (*REWARD_METRICS, *TRACKING_ERROR_METRICS)]
        columns.extend((
            term, done, diagnostics["height"], diagnostics["tilt"],
            done & ~term,
            transition["progress"] * done,
        ))
        values = torch.stack(columns, dim=-1).detach()
        self.sums.add_(values.sum(0))
        self.samples += term.numel()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        values = self.sums.tolist()
        reward_count = len(REWARD_METRICS)
        result = {
            name: value / self.samples / self.max_step_reward
            for name, value in zip(REWARD_METRICS, values[:reward_count], strict=True)
        }
        error_end = reward_count + len(TRACKING_ERROR_METRICS)
        result.update({
            name: value / self.samples
            for name, value in zip(TRACKING_ERROR_METRICS, values[reward_count:error_end], strict=True)
        })
        failures, ended, height, tilt, successes, progress_sum = values[error_end:]
        fraction = lambda count: count / ended if ended else math.nan
        result.update(
            success_rate=fraction(successes),
            progress=fraction(progress_sum),
            term_frac=fraction(failures),
            term_height_frac=fraction(height),
            term_tilt_frac=fraction(tilt),
        )
        return result

    def reset(self) -> None:
        self.sums.zero_()
        self.samples = 0


def prepare_training_log(path) -> tuple[str, ...]:
    """Extend CSV headers atomically and preserve historical columns of retired metrics."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerow(TRAIN_METRICS)
        return TRAIN_METRICS
    temporary = None
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not set(reader.fieldnames) <= set(TRAIN_METRICS + RETIRED_TRAIN_METRICS):
                raise ValueError(f"Cannot append training metrics to unknown CSV schema: {path}")
            columns = TRAIN_METRICS + tuple(name for name in reader.fieldnames if name in RETIRED_TRAIN_METRICS)
            if tuple(reader.fieldnames) == columns:
                return columns
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False,
            ) as output:
                temporary = Path(output.name)
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(reader)
        os.replace(temporary, path)
        return columns
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def group_training_metrics(metrics: dict) -> dict:
    """Report observed scalars only; iteration is already the logging step."""
    return {
        f"{group}/{label}": metrics[name]
        for group, names in SWANLAB_METRIC_GROUPS.items()
        for name, label in names.items()
        if metrics.get(name) is not None and math.isfinite(metrics[name])
    }


def create_training_logger(cfg, config_data: dict):
    """Return a run context that flushes metrics on success, failure or interruption."""
    if cfg.logger.backend == "none" or cfg.logger.mode == "disabled":
        return nullcontext(None)

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env", override=False)

    import swanlab

    return swanlab.init(
        project=cfg.logger.project,
        workspace=cfg.logger.workspace,
        experiment_name=cfg.run_name,
        config=config_data,
        logdir=str(Path(cfg.log_dir) / "swanlab"),
        mode=cfg.logger.mode,
    )
