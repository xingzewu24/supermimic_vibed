"""Typed Hydra configuration; importing this module does not initialize CUDA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_DIR = str(REPO / "data/hoi/motions/test")
DEFAULT_NJMAX = 1536
DEFAULT_EPA_HORIZON = 128
CONTROL_VERSION = 5
TASK = "g1_inspire_hoi"
STATE_VERSION = 1


@dataclass
class HandControlConfig:
    action_rate_weight: float = 0.01
    velocity_weight: float = 0.0  # Legacy config/checkpoint field; ignored by the reward.
    joint_limit_weight: float = 2.0
    wm_loss_weight: float = 1.0


@dataclass
class ObjectTrackingConfig:
    # g1_hoi_learning/configs/track/train.yaml overrides the registered task defaults.
    pos_std: float = 0.2
    rot_std: float = 0.3
    pos_weight: float = 2.0
    rot_weight: float = 2.0
    wm_loss_weight: float = 1.0


@dataclass
class NetworkConfig:
    wm_hidden: list[int] = field(default_factory=lambda: [2048, 2048, 2048])
    encoder_hidden: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    decoder_hidden: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    critic_hidden: list[int] = field(default_factory=lambda: [256, 128])
    latent_dim: int = 32


@dataclass
class OptimizerConfig:
    wm_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    pi_betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    critic_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.0
    wm_grad_norm: float = 1.0
    pi_grad_norm: float = 1.0
    pi_grad_value: float = 0.1
    critic_grad_norm: float = 1.0
    lr_final_factor: float = 0.03


@dataclass
class LoggerConfig:
    backend: str = "swanlab"
    project: str = "supermimic"
    workspace: str | None = None
    mode: str = "online"


@dataclass
class TrainConfig:
    run_name: str = "run"
    device: str = "cuda:0"
    num_envs: int = 4096
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    hist: int = 10
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_every: int = 50
    motion_dir: str = DEFAULT_MOTION_DIR
    motion_clips: list[str] | None = None
    robot_model: str = "g1_inspire"
    control_version: int = CONTROL_VERSION
    task: str = TASK
    state_version: int = STATE_VERSION
    motion_signature: str | None = None
    hand: HandControlConfig = field(default_factory=HandControlConfig)
    object: ObjectTrackingConfig = field(default_factory=ObjectTrackingConfig)
    log_dir: str = str(REPO / "logs" / "run")
    seed: int = 0
    resume: str | None = None
    init_from: str | None = None
    amp: str = "bf16"
    tf32: bool = True
    no_compile: bool = False
    njmax: int = DEFAULT_NJMAX
    nconmax: int = 128
    epa_horizon: int = DEFAULT_EPA_HORIZON
    headless: bool = False
    profile: bool = False
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class EvalConfig:
    run_dir: str = MISSING
    device: str = "cuda:0"
    watch: bool = False
    ckpt: str | None = None
    record: str | None = None
    motion_dir: str | None = None
    every: int = 1
    njmax: int = DEFAULT_NJMAX
    nconmax: int = 128
    epa_horizon: int = DEFAULT_EPA_HORIZON
    headless: bool = False


@dataclass
class PlayConfig:
    run_dir: str = MISSING
    ckpt: str | None = None
    device: str = "cuda:0"
    motion_dir: str | None = None
    clip: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    fps: int = 50
    speed: float = 1.0
    paused: bool = False
    loop: bool = True
    show_reference: bool = True
    align_reference: bool = False
    reference_opacity: float = 0.3
    njmax: int = DEFAULT_NJMAX
    nconmax: int = 128
    epa_horizon: int = DEFAULT_EPA_HORIZON


def register_configs() -> None:
    OmegaConf.register_new_resolver("repo", lambda: str(REPO), replace=True)
    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=TrainConfig)
    cs.store(name="eval_schema", node=EvalConfig)
    cs.store(name="play_schema", node=PlayConfig)


def validate_play_config(cfg: DictConfig) -> None:
    if not 1 <= cfg.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if not 1 <= cfg.fps <= 60:
        raise ValueError("fps must be in [1, 60]")
    if not math.isfinite(cfg.speed) or not 0.1 <= cfg.speed <= 4:
        raise ValueError("speed must be in [0.1, 4]")
    if not math.isfinite(cfg.reference_opacity) or not 0 <= cfg.reference_opacity <= 1:
        raise ValueError("reference_opacity must be in [0, 1]")
    for key in ("njmax", "nconmax"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg.epa_horizon < 24:
        raise ValueError("epa_horizon must be at least 24")


def save_train_config(cfg: DictConfig, path: str | Path) -> None:
    """Save a reusable snapshot, including Hydra's output-directory policy."""
    snapshot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    snapshot.defaults = ["train_schema", "_self_"]
    snapshot.hydra = {
        "job": {"chdir": False},
        "run": {"dir": "${log_dir}"},
        "sweep": {
            "dir": "${log_dir}/multirun/${now:%Y-%m-%d_%H-%M-%S}",
            "subdir": "${hydra.job.num}",
        },
    }
    OmegaConf.save(snapshot, path)


def validate_train_config(cfg: DictConfig) -> None:
    """Reject invalid sampling/update settings before loading assets or initializing CUDA."""
    validate_control_checkpoint(OmegaConf.to_container(cfg, resolve=True))
    if cfg.resume and cfg.init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    if cfg.motion_clips is not None and (not cfg.motion_clips or len(set(cfg.motion_clips)) != len(cfg.motion_clips)):
        raise ValueError("motion_clips must contain unique clip stems, or be null for all clips")
    validate_hand_config(cfg.hand)
    validate_object_config(cfg.object)
    if cfg.epa_horizon < 24:
        raise ValueError("epa_horizon must be at least 24")
    if cfg.robot_model != "g1_inspire":
        raise ValueError("robot_model must be g1_inspire")
    positive = (
        "num_envs",
        "iterations",
        "buffer_steps",
        "horizon",
        "goal_frames",
        "hist",
        "wm_window",
        "wm_batches",
        "wm_batch",
        "pi_batch",
        "critic_epochs",
        "critic_minibatches",
        "lr_decay_iters",
        "ckpt_every",
        "log_every",
        "njmax",
        "nconmax",
    )
    for key in positive:
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("warmup_steps", "wm_only_iters"):
        if cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("lr", "utd"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= cfg.gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if not 0 <= cfg.lam <= 1:
        raise ValueError("lam must be in [0, 1]")
    if not math.isfinite(cfg.action_noise) or cfg.action_noise < 0:
        raise ValueError("action_noise must be finite and nonnegative")
    if cfg.amp not in ("none", "bf16"):
        raise ValueError("amp must be 'none' or 'bf16'")
    if cfg.logger.backend not in ("swanlab", "none"):
        raise ValueError("logger.backend must be 'swanlab' or 'none'")
    if cfg.logger.mode not in ("online", "offline", "disabled"):
        raise ValueError("logger.mode must be 'online', 'offline' or 'disabled'")
    if not cfg.logger.project.strip():
        raise ValueError("logger.project must not be empty")
    if not (cfg.device == "cuda" or cfg.device.startswith("cuda:")):
        raise ValueError("training requires a CUDA device, e.g. device=cuda:0")
    if min(cfg.buffer_steps, cfg.warmup_steps + int(1 / cfg.utd)) < cfg.wm_window:
        raise ValueError("buffer_steps and initial collected steps must cover wm_window")
    if not cfg.no_critic and cfg.pi_batch * cfg.horizon < cfg.critic_minibatches:
        raise ValueError("pi_batch * horizon must be at least critic_minibatches")
    for key in ("wm_hidden", "encoder_hidden", "decoder_hidden", "critic_hidden"):
        widths = cfg.network[key]
        if not widths or any(w <= 0 for w in widths):
            raise ValueError(f"network.{key} must contain positive widths")
    if cfg.network.latent_dim <= 0:
        raise ValueError("network.latent_dim must be positive")
    opt = cfg.optimizer
    for key in ("wm_betas", "pi_betas", "critic_betas"):
        if len(opt[key]) != 2 or any(not 0 <= b < 1 for b in opt[key]):
            raise ValueError(f"optimizer.{key} must contain two values in [0, 1)")
    for key in ("eps", "wm_grad_norm", "pi_grad_norm", "pi_grad_value", "critic_grad_norm"):
        if not math.isfinite(opt[key]) or opt[key] <= 0:
            raise ValueError(f"optimizer.{key} must be finite and positive")
    if not math.isfinite(opt.weight_decay) or opt.weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be finite and nonnegative")
    if not 0 < opt.lr_final_factor <= 1:
        raise ValueError("optimizer.lr_final_factor must be in (0, 1]")


def checkpoint_network_config(saved_args: dict) -> dict:
    """Old argparse checkpoints used the original, fixed network architecture."""
    cfg = OmegaConf.merge(OmegaConf.structured(NetworkConfig), saved_args.get("network", {}))
    return OmegaConf.to_container(cfg, resolve=True)


def validate_resume_config(cfg: DictConfig, saved_args: dict) -> None:
    if saved_args.get("robot_model") != "g1_inspire":
        raise ValueError("resume requires a G1 Inspire checkpoint; the old body's world model is not equivalent")
    validate_control_checkpoint(saved_args)
    active_hand = OmegaConf.to_container(cfg.hand, resolve=True)
    saved_hand = dict(saved_args.get("hand") or {})
    active_hand.pop("velocity_weight", None)
    saved_hand.pop("velocity_weight", None)
    if active_hand != saved_hand:
        raise ValueError("resume requires the checkpoint's hand configuration")
    if OmegaConf.to_container(cfg.object, resolve=True) != saved_args.get("object"):
        raise ValueError("resume requires the checkpoint's object configuration")
    for key in ("hist", "goal_frames"):
        if cfg[key] != saved_args[key]:
            raise ValueError(f"resume requires {key}={saved_args[key]} from the checkpoint")
    if OmegaConf.to_container(cfg.network, resolve=True) != checkpoint_network_config(saved_args):
        raise ValueError("resume requires the checkpoint's network configuration")


def validate_hand_config(cfg) -> None:
    for key in ("wm_loss_weight",):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f"hand.{key} must be finite and positive")
    for key in ("action_rate_weight", "joint_limit_weight"):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f"hand.{key} must be finite and nonnegative")


def validate_control_checkpoint(saved_args: dict) -> None:
    if (saved_args.get("control_version"), saved_args.get("task"), saved_args.get("state_version")) != (
        CONTROL_VERSION, TASK, STATE_VERSION,
    ):
        raise ValueError(f"Requires a HOI checkpoint (control_version={CONTROL_VERSION}, task={TASK}, "
                         f"state_version={STATE_VERSION}) with all 24 hand actuators and aligned object rewards. "
                         "Use init_from for matching-layer initialization from an older checkpoint.")


def validate_object_config(cfg) -> None:
    for key in ("pos_std", "rot_std", "wm_loss_weight"):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f"object.{key} must be finite and positive")
    for key in ("pos_weight", "rot_weight"):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f"object.{key} must be finite and nonnegative")
