"""Hydra migration contracts: defaults, paper values, snapshots and legacy checkpoints."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir, initialize_config_module
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from supermimic.config import (
    CONTROL_VERSION,
    REPO,
    NetworkConfig,
    TrainConfig,
    checkpoint_network_config,
    register_configs,
    save_train_config,
    validate_control_checkpoint,
    validate_play_config,
    validate_resume_config,
    validate_train_config,
)

register_configs()


@pytest.mark.parametrize("old_version", [2, 3, 4])
def test_old_control_and_reward_versions_cannot_resume_silently(old_version):
    args = {"control_version": old_version, "task": "g1_inspire_hoi", "state_version": 1}
    with pytest.raises(ValueError, match=f"control_version={CONTROL_VERSION}"):
        validate_control_checkpoint(args)


def configuration(name="train", overrides=()):
    with initialize_config_module(config_module="supermimic.conf", version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def test_existing_codebase_defaults_are_preserved(monkeypatch):
    monkeypatch.delenv("SM_PROFILE", raising=False)
    actual = OmegaConf.to_container(configuration(), resolve=True)
    # Snapshot of every pre-Hydra argparse option, independent of the new schema.
    previous = {
        "num_envs": 4096,
        "iterations": 320000,
        "buffer_steps": 1000,
        "utd": 0.4,
        "horizon": 32,
        "goal_frames": 10,
        "hist": 10,
        "wm_window": 8,
        "wm_batches": 10,
        "wm_batch": 2048,
        "pi_batch": 4096,
        "lr": 0.0002,
        "gamma": 0.99,
        "lam": 0.95,
        "action_noise": 0.1,
        "critic_epochs": 5,
        "critic_minibatches": 4,
        "no_critic": False,
        "warmup_steps": 64,
        "wm_only_iters": 500,
        "lr_decay_iters": 250000,
        "ckpt_every": 2000,
        "log_every": 50,
        "seed": 0,
        "resume": None,
        "amp": "bf16",
        "device": "cuda:0",
        "njmax": 1536,
        "nconmax": 128,
        "headless": False,
        "no_compile": False,
        "motion_dir": str(REPO / "data/hoi/motions/test"),
        "log_dir": str(REPO / "logs/run"),
    }
    assert {key: actual[key] for key in previous} == previous
    assert actual["logger"] == {
        "backend": "swanlab", "project": "supermimic", "workspace": None, "mode": "online",
    }
    validate_train_config(configuration())


def test_template_explicitly_lists_every_parameter():
    template = OmegaConf.load(REPO / "supermimic/conf/train.yaml")
    schema = OmegaConf.structured(TrainConfig)
    assert set(template) - {"defaults", "hydra"} == set(schema)
    for group in ("network", "optimizer", "logger", "hand", "object"):
        assert set(template[group]) == set(schema[group])


def test_object_rewards_use_reference_training_defaults_and_finger_tracking_is_removed():
    cfg = configuration()
    assert (cfg.object.pos_weight, cfg.object.rot_weight) == (2.0, 2.0)
    assert (cfg.object.pos_std, cfg.object.rot_std) == (0.2, 0.3)
    assert "track_weight" not in cfg.hand and "track_std" not in cfg.hand
    with pytest.raises(ConfigCompositionException):
        configuration(overrides=["hand.track_weight=1.0"])


def test_play_defaults_and_validation():
    cfg = configuration("play", ["run_dir=/tmp/example_run"])
    assert cfg.host == "127.0.0.1" and cfg.port == 8080
    assert cfg.ckpt is None and cfg.motion_dir is None and cfg.clip is None
    assert cfg.speed == 1.0 and cfg.loop
    assert cfg.show_reference and not cfg.align_reference and cfg.reference_opacity == 0.3
    validate_play_config(cfg)


@pytest.mark.parametrize("override", [
    "port=0", "port=65536", "fps=0", "fps=61", "speed=0", "speed=nan",
    "reference_opacity=-0.1", "reference_opacity=1.1", "reference_opacity=nan",
])
def test_invalid_play_settings_fail_before_cuda(override):
    with pytest.raises(ValueError):
        validate_play_config(configuration("play", ["run_dir=/tmp/example_run", override]))


def test_paper_single_gpu_published_hyperparameters():
    cfg = configuration("paper")
    expected = {
        "wm_window": 8,
        "wm_batches": 10,
        "wm_batch": 2048,
        "horizon": 32,
        "pi_batch": 4096,
        "lr": 0.0002,
        "gamma": 0.99,
        "lam": 0.95,
        "critic_epochs": 5,
        "critic_minibatches": 4,
        "num_envs": 4096,
        "action_noise": 0.1,
        "buffer_steps": 1000,
        "utd": 0.4,
        "lr_decay_iters": 250000,
        "iterations": 320000,
        "hist": 10,
        "goal_frames": 10,
        "no_critic": False,
        "amp": "none",
        "tf32": True,
        "wm_only_iters": 0,
    }
    assert {key: cfg[key] for key in expected} == expected
    assert list(cfg.network.wm_hidden) == [2048, 2048, 2048]
    assert list(cfg.network.encoder_hidden) == list(cfg.network.decoder_hidden) == [2048, 1024, 512, 256]
    assert list(cfg.network.critic_hidden) == [256, 128]
    assert cfg.network.latent_dim == 32
    assert list(cfg.optimizer.pi_betas) == [0.9, 0.95]
    assert cfg.optimizer.wm_grad_norm == cfg.optimizer.pi_grad_norm == cfg.optimizer.critic_grad_norm == 1.0
    assert cfg.optimizer.pi_grad_value == 0.1
    assert cfg.optimizer.lr_final_factor == 0.03
    validate_train_config(cfg)


def test_typed_overrides_and_legacy_profile_environment(monkeypatch):
    monkeypatch.setenv("SM_PROFILE", "1")
    cfg = configuration(overrides=["num_envs=16", "no_compile=true", "optimizer.pi_betas=[0.8,0.9]"])
    assert cfg.num_envs == 16 and cfg.no_compile and cfg.profile
    assert list(cfg.optimizer.pi_betas) == [0.8, 0.9]
    with pytest.raises(ConfigCompositionException):
        configuration(overrides=["num_envs=not_an_integer"])
    with pytest.raises(ConfigCompositionException):
        configuration(overrides=["num_env=16"])


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["buffer_steps=1"], "wm_window"),
        (["warmup_steps=0"], "initial collected steps"),
        (["utd=0"], "utd"),
        (["gamma=1"], "gamma"),
        (["amp=fp16"], "amp"),
        (["logger.backend=wandb"], "logger.backend"),
        (["logger.mode=invalid"], "logger.mode"),
        (["logger.project=''"], "logger.project"),
        (["pi_batch=1", "horizon=1", "critic_minibatches=4"], "critic_minibatches"),
        (["optimizer.pi_betas=[0.9,1.0]"], "pi_betas"),
        (["object.pos_std=0"], "pos_std"),
        (["object.rot_weight=-1"], "rot_weight"),
        (["hand.action_rate_weight=nan"], "action_rate_weight"),
        (["hand.wm_loss_weight=0"], "wm_loss_weight"),
        (["epa_horizon=23"], "epa_horizon"),
    ],
)
def test_invalid_training_settings_fail_before_cuda(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_train_config(configuration(overrides=overrides))


def test_resolved_snapshot_can_be_reused(tmp_path):
    cfg = configuration("smoke", ["optimizer.pi_grad_value=0.2", "network.latent_dim=16"])
    OmegaConf.resolve(cfg)
    save_train_config(cfg, tmp_path / "config.yaml")
    with initialize_config_dir(config_dir=str(tmp_path), version_base="1.3"):
        restored = compose(config_name="config", overrides=["iterations=3"])
        with_hydra = compose(config_name="config", return_hydra_config=True, overrides=["log_dir=/tmp/resumed_run"])
    assert with_hydra.hydra.run.dir == "/tmp/resumed_run"
    assert not with_hydra.hydra.job.chdir
    restored = OmegaConf.merge(OmegaConf.structured(TrainConfig), restored)
    validate_train_config(restored)
    assert restored.iterations == 3
    assert restored.hist == 3 and restored.goal_frames == 2
    assert restored.optimizer.pi_grad_value == 0.2 and restored.network.latent_dim == 16


def test_legacy_checkpoint_fallback_and_resume_dimensions():
    legacy = {"hist": 10, "goal_frames": 10, "amp": "bf16"}
    assert checkpoint_network_config(legacy) == OmegaConf.to_container(OmegaConf.structured(NetworkConfig))
    with pytest.raises(ValueError, match="G1 Inspire checkpoint"):
        validate_resume_config(configuration(), legacy)
    legacy["robot_model"] = "g1_inspire"
    with pytest.raises(ValueError, match="HOI checkpoint"):
        validate_resume_config(configuration(), legacy)
    legacy.update(OmegaConf.to_container(configuration(), resolve=True))
    validate_resume_config(configuration(), legacy)
    assert configuration().hand.velocity_weight == 0.0
    legacy["hand"]["velocity_weight"] = 0.001
    validate_resume_config(configuration(), legacy)
    del legacy["hand"]["velocity_weight"]
    validate_resume_config(configuration(), legacy)
    with pytest.raises(ValueError, match="hand configuration"):
        validate_resume_config(configuration(overrides=["hand.action_rate_weight=0.02"]), legacy)
    with pytest.raises(ValueError, match="hist=10"):
        validate_resume_config(configuration(overrides=["hist=3"]), legacy)
    with pytest.raises(ValueError, match="goal_frames=10"):
        validate_resume_config(configuration(overrides=["goal_frames=3"]), legacy)
    with pytest.raises(ValueError, match="network configuration"):
        validate_resume_config(configuration(overrides=["network.latent_dim=16"]), legacy)


def test_entrypoints_import_and_print_config_without_torch(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import supermimic.train, supermimic.evaluate, supermimic.play, sys; "
                "assert not {'torch', 'swanlab', 'dotenv', 'viser', 'mjviser'} & sys.modules.keys()"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not result.stdout
    for module, options in (
        ("supermimic.train", ["--config-name", "paper"]),
        ("supermimic.evaluate", ["run_dir=/tmp/example_run"]),
        ("supermimic.play", ["run_dir=/tmp/example_run"]),
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, *options, "--cfg", "job", "--resolve"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        cfg = OmegaConf.create(result.stdout)
        if cfg.motion_dir is not None:
            assert Path(cfg.motion_dir).is_absolute()
    assert not list(tmp_path.iterdir())
