"""Reject invalid transitions and sampling statistics before CUDA assertions occur."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from supermimic.config import DEFAULT_NJMAX, EvalConfig, PlayConfig, TrainConfig
from supermimic.env import TrackingSim, make_sim


def stub_env():
    env = TrackingSim.__new__(TrackingSim)
    env.N = 2
    env.device = "cpu"
    env.f = torch.tensor([1, 2])
    env.motion = SimpleNamespace(
        names=["test_motion"], clip_id=torch.zeros(10, dtype=torch.long),
        num_bins=2, bin_start=torch.tensor([0, 5]), bin_clip=torch.zeros(2, dtype=torch.long),
        clip_end=torch.tensor([9]),
    )
    env._overflow = torch.zeros(2, dtype=torch.int32)
    env._nefc = torch.tensor([16, 20], dtype=torch.int32)
    env._peak_nefc = torch.zeros_like(env._nefc)
    env.sim = SimpleNamespace(
        wp_data=SimpleNamespace(njmax=DEFAULT_NJMAX),
        data=SimpleNamespace(qpos=torch.zeros(2, 60), qvel=torch.zeros(2, 59),
                             qacc=torch.zeros(2, 59), qfrc_constraint=torch.zeros(2, 59)),
    )
    env.G_max = 10.0
    env.bin_G = torch.zeros(2)
    env.bin_w_avg = torch.full((2,), 0.5)
    return env


def test_all_simulation_entrypoints_use_inspire_capacity():
    assert DEFAULT_NJMAX == 1536
    assert make_sim("cpu").physics.njmax == DEFAULT_NJMAX
    for cls in (TrainConfig, EvalConfig, PlayConfig):
        assert OmegaConf.structured(cls).njmax == DEFAULT_NJMAX


def test_healthy_state_and_sampling_stay_finite():
    env = stub_env()
    env._check_physics("test")
    frames, bins = env.sample_frames(100)
    assert bool((frames >= env.motion.bin_start[bins]).all())
    assert bool((frames < env.motion.clip_end[0]).all())
    assert torch.isfinite(env.bin_w_avg).all()


def test_episode_progress_accounts_for_random_start_and_time_limit():
    env = TrackingSim.__new__(TrackingSim)
    env.max_episode_steps = 500
    env.motion = SimpleNamespace(clip_end=torch.tensor([1000]), clip_id=torch.zeros(1001, dtype=torch.long))
    # Reset at 900: 100 steps remain. Reset at 300: the 500-step cap comes first.
    frames = torch.tensor([900, 901, 999, 300, 799, 1000])
    steps = torch.tensor([0, 1, 99, 0, 499, 1000])
    torch.testing.assert_close(
        env._episode_progress(frames, steps), torch.tensor([0.01, 0.02, 1.0, 0.002, 1.0, 1.0]),
    )


def test_current_and_previous_substep_overflows_produce_actionable_errors():
    env = stub_env()
    env.sim.wp_data.njmax = 500
    env._nefc[1] = 740
    with pytest.raises(RuntimeError, match=r"njmax=500, observed nefc=740.*env_ids=\[1\].*test_motion"):
        env._check_physics("after physics substeps")
    # Later substeps can have fewer contacts; keep the peak of the whole control cycle.
    env._nefc[1] = 20
    with pytest.raises(RuntimeError, match="observed nefc=740"):
        env._check_physics("later substep")


def test_other_overflow_flags_are_not_silently_ignored():
    env = stub_env()
    env._overflow[0] = 8
    with pytest.raises(RuntimeError, match="simulation capacity overflow.*overflow flags=\\[8\\]"):
        env._check_physics("test")


def test_epa_horizon_overflow_is_not_misreported_as_insufficient_njmax():
    env = stub_env()
    env._nefc[0] = 372
    env._overflow[0] = 256
    with pytest.raises(RuntimeError, match="EPA_HORIZON.*increase epa_horizon") as error:
        env._check_physics("during forward/reset")
    assert "increase njmax" not in str(error.value)


def test_combined_epa_and_constraint_flags_report_both_capacities():
    env = stub_env()
    env._overflow[0] = 257
    with pytest.raises(RuntimeError, match="NEFC\\|EPA_HORIZON") as error:
        env._check_physics("test")
    assert "increase epa_horizon" in str(error.value) and "increase njmax" in str(error.value)


@pytest.mark.parametrize("name", ["qpos", "qvel", "qacc", "qfrc_constraint"])
def test_nonfinite_physics_reports_the_failing_world(name):
    env = stub_env()
    getattr(env.sim.data, name)[1, 0] = float("nan")
    with pytest.raises(RuntimeError, match=r"non-finite physics state.*env_ids=\[1\]"):
        env._check_physics("test")


@pytest.mark.parametrize("field", ["bin_G", "bin_w_avg"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_sampling_never_reaches_multinomial(field, invalid, monkeypatch):
    env = stub_env()
    getattr(env, field)[0] = invalid

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid probabilities must not reach a CUDA kernel")

    monkeypatch.setattr(torch, "multinomial", forbidden)
    with pytest.raises(RuntimeError, match="Non-finite motion sampling statistics"):
        env.sample_frames(1)


def test_invalid_probabilities_do_not_update_sampling_history(monkeypatch):
    env = stub_env()
    env.bin_w_avg[:] = torch.tensor([-10.0, 11.0])
    before = env.bin_w_avg.clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid probabilities must not reach a CUDA kernel")

    monkeypatch.setattr(torch, "multinomial", forbidden)
    with pytest.raises(RuntimeError, match="Invalid motion sampling probabilities"):
        env.sample_frames(1)
    torch.testing.assert_close(env.bin_w_avg, before, atol=0, rtol=0)


@pytest.mark.parametrize("label,shape", [("policy action", (2, 29)), ("tracking reward", (2,))])
def test_nonfinite_actions_and_rewards_report_the_world(label, shape):
    env = stub_env()
    values = torch.zeros(shape)
    values[0] = float("inf")
    with pytest.raises(RuntimeError, match=r"Non-finite .*env_ids=\[0\]"):
        env._ensure_finite(values, label)
