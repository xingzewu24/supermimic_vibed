"""Integration checks for the simulator migration (small synthetic reference, real G1)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, body_state, unflatten_state
from supermimic.motion import MotionLib
from supermimic.robot import (
    ALL_HAND_JOINT_NAMES,
    G1_JOINT_NAMES,
    MJCF_PATH,
    _kd,
    _kp,
    default_joint_pos_list,
    motor_of,
)
from supermimic.rotations import quat_to_mat


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    if not Path(MJCF_PATH).exists():
        pytest.skip("run scripts/download_data.sh to build the G1 MJCF")
    device = "cuda:0"
    torch.cuda.set_device(device)
    kin = G1Kinematics(device)
    from supermimic.config import DEFAULT_MOTION_DIR

    q0 = torch.tensor(default_joint_pos_list(), device=device)
    motion = MotionLib.from_dir(DEFAULT_MOTION_DIR, kin, device)
    return motion, MDP(kin, motion, q0)


@pytest.fixture(scope="module")
def env(reference):
    motion, mdp = reference
    return TrackingSim(make_sim("cuda:0"), 4, motion, mdp, eval_mode=True)


def assert_state(actual, expected):
    if actual[4].shape[-1] == 53 and expected[4].shape[-1] == 29:
        actual = body_state(actual)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)


def test_joint_drives_and_collision_masks(env):
    model = env.sim.mj_model
    assert (model.nq, model.nv, model.nu, model.neq) == (67, 65, 53, 12)
    for name in G1_JOINT_NAMES:
        joint = model.joint(f"robot/{name}")
        aid = np.flatnonzero(model.actuator_trnid[:, 0] == joint.id).item()
        effort, _, armature = motor_of(name)
        assert model.actuator_gainprm[aid, 0] == pytest.approx(_kp(armature))
        assert model.actuator_biasprm[aid, 2] == pytest.approx(-_kd(armature))
        assert model.dof_armature[joint.dofadr[0]] == pytest.approx(armature)
        np.testing.assert_allclose(model.actuator_forcerange[aid], [-effort, effort])
        np.testing.assert_allclose(model.jnt_actfrcrange[joint.id], [-effort, effort])
    ids = env.robot.indexing.geom_ids.cpu().numpy()
    assert (model.geom_contype[ids] == 0).all()
    assert (model.geom_conaffinity[ids] == 5).any()


def test_wxyz_world_velocities_and_joint_order_roundtrip(env):
    ids = torch.arange(env.N, device=env.device)
    frames = torch.tensor([20, 70, 150, 250], device=env.device)
    env.reset(ids, frames)
    assert_state(env.state(), env.motion.ref_state(frames))
    native_q = env.robot.data.joint_pos[:, env.joint_ids]
    torch.testing.assert_close(native_q, env.motion.q[frames])


def test_partial_reset_preserves_other_worlds_and_replay(env):
    env.step(torch.zeros(env.N, 41, device=env.device))
    before = env.state()
    ids = torch.tensor([0, 2], device=env.device)
    others = torch.tensor([1, 3], device=env.device)
    frames = torch.tensor([60, 170], device=env.device)
    env.reset(ids, frames)
    assert_state(tuple(s[ids] for s in env.state()), env.motion.ref_state(frames))
    for actual, old in zip(env.state(), before):
        torch.testing.assert_close(actual[others], old[others], atol=0, rtol=0)
    out = env.step(torch.zeros(env.N, 41, device=env.device))
    assert_state(tuple(s[ids] for s in unflatten_state(out["s"])), env.motion.ref_state(frames))


def test_fk_and_integrated_state_after_four_substeps(env):
    before = env.sim.data.time.clone()
    out = env.step(torch.zeros(env.N, 41, device=env.device))
    torch.testing.assert_close(env.sim.data.time - before, torch.full_like(before, 0.02), atol=1e-6, rtol=1e-5)
    assert_state(unflatten_state(out["s1"]), env.state())
    env.refresh()
    bp, bR, bv, bw = env.mdp.kin.forward(*env.state()[:6], all_bodies=True)
    ids = [env.robot.body_names.index(name) for name in env.mdp.kin.body_names]
    pose = env.robot.data.body_link_pose_w[:, ids]
    velocity = env.robot.data.body_link_vel_w[:, ids]
    for actual, expected in (
        (pose[..., :3], bp),
        (quat_to_mat(pose[..., 3:]), bR),
        (velocity[..., :3], bv),
        (velocity[..., 3:], bw),
    ):
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


def test_finished_clip_stays_in_bounds(env):
    frames = env.motion.clip_end[torch.arange(env.N, device=env.device) % len(env.motion.names)]
    env.reset(torch.arange(env.N, device=env.device), frames)
    for _ in range(3):
        out = env.step(torch.zeros(env.N, 41, device=env.device))
        assert out["done"].all()
        assert torch.isfinite(out["r"]).all()
        torch.testing.assert_close(env.f, frames)


def test_randomization_and_training_autoreset(reference):
    motion, mdp = reference
    env = TrackingSim(make_sim("cuda:0"), 4, motion, mdp, max_episode_steps=2, reset_noise=False)
    assert ((env.xi[:, 0] >= 0.3) & (env.xi[:, 0] <= 1.2)).all()
    assert env.xi[:, 0].std() > 0
    assert (env.xi[:, 1:30].abs() <= 0.01).all()
    geom_ids = env.robot.indexing.geom_ids
    actual = env.sim.model.geom_friction[:, geom_ids, 0]
    torch.testing.assert_close(actual, env.xi[:, :1].expand_as(actual))
    torso = env.robot.indexing.body_ids[env.torso_id]
    default = env.sim.get_default_field("body_ipos")[torso]
    torch.testing.assert_close(env.sim.model.body_ipos[:, torso] - default, env.xi[:, 30:], atol=1e-7, rtol=1e-5)
    # set_const writes these derived constants; shared arrays would race across worlds.
    assert env.sim.model.body_invweight0.shape[0] == env.N
    assert env.sim.model.dof_invweight0.shape[0] == env.N
    assert env.sim.model.actuator_acc0.shape[0] == env.N
    start_xi = env.xi.clone()
    for _ in range(2):
        out = env.step(torch.zeros(env.N, 41, device=env.device))
        assert torch.isfinite(out["s1"]).all()
    assert out["done"].all()
    assert (env.ep_step == 0).all()
    torch.testing.assert_close(env.xi[:, 0], start_xi[:, 0])
    torch.testing.assert_close(env.xi[:, 30:], start_xi[:, 30:])
    assert not torch.equal(env.xi[:, 1:30], start_xi[:, 1:30])


def test_motion_hand_pose_control_and_partial_reset(reference):
    motion, mdp = reference
    env = TrackingSim(make_sim("cuda:0"), 4, motion, mdp, eval_mode=True)
    ids = torch.arange(4, device=env.device)
    frames = torch.tensor([10, 80, 210, 350], device=env.device)
    env.reset(ids, frames)
    torch.testing.assert_close(env.robot.data.joint_pos[:, env.all_hand_joint_ids], motion.hand_q[frames])
    torch.testing.assert_close(env.robot.data.joint_vel[:, env.all_hand_joint_ids], motion.hand_qd[frames])
    action = torch.zeros(4, 41, device=env.device)
    action[:, 29:] = 0.3
    out = env.step(action)
    model = env.sim.mj_model
    act_ids = [int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(f"robot/{n}").id)[0])
               for n in ALL_HAND_JOINT_NAMES]
    torch.testing.assert_close(env.sim.data.ctrl[:, act_ids], env.hand.expand(env.hand.action_target(action[:, 29:])))
    torch.testing.assert_close(out["hand_reference"], motion.hand_reference(frames + 1))
    assert out["a"].shape == (4, 41) and out["s1"].shape == (4, 132)
    # Simulator transition labels and imagination must use exactly the same reward head.
    predicted_reward = mdp.reward(unflatten_state(out["s1"]), frames + 1, action, out["a_prev"])
    # The replay quaternion roundtrip introduces float32 rotation rounding.
    torch.testing.assert_close(out["r"], predicted_reward, atol=1e-7, rtol=1e-6)
    assert "reward_hand_track" not in out["terms"]
    assert not hasattr(env, "set_hand_command") and not hasattr(env.hand, "command")


@pytest.mark.parametrize("capacity", [500, 1536])
def test_dense_ground_contacts_fail_cleanly_or_step_with_sufficient_capacity(reference, capacity, monkeypatch):
    from types import SimpleNamespace

    motion, mdp = reference
    # A synthetic deeply penetrating pose exercises the same hand/ground contact peak
    # as floor-level motion resets, without depending on a particular motion dataset.
    cfg = make_sim("cuda:0", SimpleNamespace(njmax=capacity, nconmax=512))
    env = TrackingSim(cfg, 4, motion, mdp, eval_mode=True)
    pose = env.sim.data.qpos[:1, env.robot.indexing.free_joint_q_adr].clone()
    pose[:, 2] = -2.0
    env.robot.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
    before_bins = env.bin_G.clone()
    action = torch.zeros(4, 41, device=env.device)
    if capacity == 500:
        def forbidden(*args, **kwargs):
            raise AssertionError("overflow must fail before reward and return updates")

        monkeypatch.setattr(mdp, "reward", forbidden)
        with pytest.raises(RuntimeError, match=r"simulation capacity overflow.*njmax=500"):
            env.step(action)
        torch.testing.assert_close(env.bin_G, before_bins, atol=0, rtol=0)
        assert torch.isfinite(env.ep_ret).all()
        # A Python error leaves the CUDA context usable for inspection or a new environment.
        assert int(torch.ones(1, device=env.device).sum()) == 1
    else:
        out = env.step(action)
        assert int(env._peak_nefc.max()) > 500
        assert torch.isfinite(out["s1"]).all() and torch.isfinite(out["r"]).all()
