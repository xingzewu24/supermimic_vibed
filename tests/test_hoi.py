"""HOI data contracts, coordinate invariance, differentiable rewards and coupled dynamics."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.checkpoint import initialize_matching_parameters
from supermimic.config import DEFAULT_MOTION_DIR
from supermimic.hoi import GOAL_FRAME_DIM, com_to_link_velocity
from supermimic.kinematics import G1Kinematics
from supermimic.logger import REWARD_METRICS
from supermimic.mdp import (
    ACTION_DIM,
    CRITIC_EXTRA,
    INV_DIM,
    MDP,
    OBS_DIM,
    STATE_DIM,
    flatten_state,
    invariant_features,
    unflatten_state,
)
from supermimic.motion import MotionLib
from supermimic.networks import WM_OUT, Policy, WorldModel, grouped_state_error
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import so3_exp

REFERENCE_CLIPS = ["sub16_largebox_012_v00", "sub16_largebox_022_v00", "sub16_largebox_038_v00"]


@pytest.fixture(scope="module")
def reference():
    files = [Path(DEFAULT_MOTION_DIR) / f"{name}.npz" for name in REFERENCE_CLIPS]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    kin = G1Kinematics("cpu")
    motion = MotionLib.from_dir(DEFAULT_MOTION_DIR, kin, "cpu", clips=REFERENCE_CLIPS)
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files} == before
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
    return motion, mdp


def test_loader_preserves_named_joints_velocities_and_raw_files(reference):
    motion, _ = reference
    files = [Path(DEFAULT_MOTION_DIR) / f"{name}.npz" for name in REFERENCE_CLIPS]
    manifest = json.loads((Path(DEFAULT_MOTION_DIR).parents[1] / "manifest.json").read_text())
    assert motion.clip_len.tolist() == [477, 414, 385]
    assert motion.num_frames == 1276
    assert sum(len(r["frames"]) for c in motion.validation["clips"] for r in c["limit_violations"]) == 263
    from supermimic.robot import ALL_HAND_JOINT_NAMES, G1_JOINT_NAMES

    for i, file in enumerate(files):
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        raw = dict(np.load(file, allow_pickle=False))
        names = raw["joint_names"].tolist() if "joint_names" in raw else manifest["joint_names"]
        index = [names.index(n) for n in G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES]
        start, end = motion.clip_start[i], motion.clip_end[i] + 1
        torch.testing.assert_close(motion.full_q[start:end], torch.tensor(raw["joint_pos"][:, index]))
        torch.testing.assert_close(motion.full_qd[start:end], torch.tensor(raw["joint_vel"][:, index]))
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
        assert max(motion.validation["clips"][i]["fk_max_error"].values()) < 3e-5
        assert max(motion.validation["clips"][i]["full_fk_max_error"].values()) < 1e-4


def test_loader_rejects_incorrect_reference_finger_poses(tmp_path, reference):
    source = Path(DEFAULT_MOTION_DIR).parents[1]
    manifest = json.loads((source / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "objects").symlink_to(source / "objects", target_is_directory=True)
    raw = dict(np.load(Path(DEFAULT_MOTION_DIR) / f"{REFERENCE_CLIPS[0]}.npz", allow_pickle=False))
    names = raw["body_names"].tolist() if "body_names" in raw else manifest["body_names"]
    raw["body_pos_w"][0, names.index("L_thumb_distal"), 0] += 0.01
    np.savez(tmp_path / "bad_finger.npz", **raw)
    with pytest.raises(ValueError, match="reference FK mismatch.*including hands"):
        MotionLib.from_dir(str(tmp_path), reference[1].kin, "cpu")


@pytest.mark.parametrize("embedded", [(), ("joint_names",), ("body_names",), ("joint_names", "body_names")])
def test_names_can_be_stored_in_manifest_or_each_motion(tmp_path, reference, embedded):
    source = Path(DEFAULT_MOTION_DIR).parents[1]
    manifest = json.loads((source / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "objects").symlink_to(source / "objects", target_is_directory=True)
    file = Path(DEFAULT_MOTION_DIR) / f"{REFERENCE_CLIPS[0]}.npz"
    raw = dict(np.load(file, allow_pickle=False))
    for key in ("joint_names", "body_names"):
        if key in embedded:
            raw[key] = np.asarray(manifest[key])
        else:
            raw.pop(key, None)
    np.savez(tmp_path / file.name, **raw)
    before = (tmp_path / file.name).read_bytes()
    loaded = MotionLib.from_dir(str(tmp_path), reference[1].kin, "cpu")
    torch.testing.assert_close(loaded.full_q, reference[0].full_q[:477])
    torch.testing.assert_close(loaded.body_p, reference[0].body_p[:477])
    assert loaded.validation["clips"][0]["name_sources"] == {
        key: "motion" if key in embedded else "manifest" for key in ("joint_names", "body_names")
    }
    assert (tmp_path / file.name).read_bytes() == before


def test_embedded_names_take_precedence_and_missing_metadata_fails_clearly(tmp_path, reference):
    source = Path(DEFAULT_MOTION_DIR).parents[1]
    manifest = json.loads((source / "manifest.json").read_text())
    (tmp_path / "objects").symlink_to(source / "objects", target_is_directory=True)
    raw = dict(np.load(Path(DEFAULT_MOTION_DIR) / f"{REFERENCE_CLIPS[0]}.npz", allow_pickle=False))
    # Explicit file metadata allows a different column order from the shared manifest.
    for key in ("joint_names", "body_names"):
        raw[key] = np.asarray(manifest[key][::-1])
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                "body_ang_vel_w", "contact_label"):
        raw[key] = raw[key][:, ::-1].copy()
    path = tmp_path / "clip.npz"
    np.savez(path, **raw)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded = MotionLib.from_dir(str(tmp_path), reference[1].kin, "cpu")
    torch.testing.assert_close(loaded.full_q, reference[0].full_q[:477])
    torch.testing.assert_close(loaded.body_p, reference[0].body_p[:477])
    del raw["joint_names"], manifest["joint_names"]
    np.savez(path, **raw)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing joint_names in both"):
        MotionLib.from_dir(str(tmp_path), reference[1].kin, "cpu")


def test_com_origin_conversion_has_correct_sign_with_rotation():
    R = so3_exp(torch.tensor([0.0, 0.0, np.pi / 2]))
    omega = torch.tensor([0.0, 0.0, 2.0])
    # COM is +y in world: v_com = v_origin + omega cross (0,1,0).
    v = com_to_link_velocity(torch.tensor([-1.0, 0, 0]), omega, R, torch.tensor([1.0, 0, 0]))
    torch.testing.assert_close(v, torch.tensor([1.0, 0, 0]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("damage", ["fps", "quaternion", "mimic", "fk", "length", "nonfinite"])
def test_invalid_hoi_is_rejected_before_simulation(tmp_path, reference, damage):
    root = Path(DEFAULT_MOTION_DIR).parents[1]
    (tmp_path / "manifest.json").write_text((root / "manifest.json").read_text())
    (tmp_path / "objects").symlink_to(root / "objects", target_is_directory=True)
    file = min(Path(DEFAULT_MOTION_DIR).glob("*.npz"))
    raw = dict(np.load(file, allow_pickle=False))
    manifest = json.loads((root / "manifest.json").read_text())
    for key in ("joint_names", "body_names"):
        raw.setdefault(key, np.array(manifest[key]))
    if damage == "fps":
        raw["fps"][:] = 30
    elif damage == "quaternion":
        raw["object_quat_w"][0] = 0
    elif damage == "mimic":
        raw["joint_pos"][0, raw["joint_names"].tolist().index("L_thumb_distal_joint")] += 0.1
    elif damage == "fk":
        raw["body_pos_w"][0, raw["body_names"].tolist().index("left_knee_link"), 0] += 0.1
    elif damage == "length":
        raw["motion_lengths"][0] -= 1
    else:
        raw["joint_vel"][0, 0] = np.nan
    np.savez(tmp_path / file.name, **raw)
    with pytest.raises(ValueError, match="Invalid HOI motion"):
        MotionLib.from_dir(str(tmp_path), reference[1].kin, "cpu")


def test_future_goals_and_replay_reference_stop_at_each_clip(reference):
    m, _ = reference
    f = torch.stack((m.clip_end[0] - 1, m.clip_end[1], m.clip_start[2]))
    future = m.future_frames(f, 10)
    assert (future[:2] == m.clip_end[:2, None]).all()
    goal = m.goal(f, 10).view(3, 10, GOAL_FRAME_DIM)
    torch.testing.assert_close(goal[..., 73:85], m.hand_reference(future))
    torch.testing.assert_close(goal[..., 85:97], m.hand_qd[future][..., :12] * 0.1)


def test_single_clip_selection_has_its_own_indices_and_signature(reference):
    m, mdp = reference
    selected = MotionLib.from_dir(DEFAULT_MOTION_DIR, mdp.kin, "cpu", clips=[m.names[1]])
    assert selected.names == [m.names[1]] and selected.clip_start.tolist() == [0]
    assert selected.clip_len.tolist() == [414] and selected.signature != m.signature
    torch.testing.assert_close(selected.hand_q, m.hand_q[477:891])
    with pytest.raises(ValueError, match="missing clips"):
        MotionLib.from_dir(DEFAULT_MOTION_DIR, mdp.kin, "cpu", clips=["missing_motion"])


def transform_state(S, yaw, shift):
    rot = lambda x: (yaw @ x.unsqueeze(-1)).squeeze(-1)
    p, R, v, w, q, qd, op, oR, ov, ow = S
    return rot(p) + shift, yaw @ R, rot(v), rot(w), q, qd, rot(op) + shift, yaw @ oR, rot(ov), rot(ow)


def test_joint_state_and_wm_roundtrip_and_common_heading_invariance(reference):
    m, _ = reference
    S, S1 = m.ref_state(torch.tensor([90, 600])), m.ref_state(torch.tensor([91, 601]))
    assert flatten_state(S).shape == (2, STATE_DIM) == (2, 132)
    for a, b in zip(unflatten_state(flatten_state(S)), S, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    wm = WorldModel(hidden=(16,))
    assert wm.in_dim == 208 and WM_OUT == 130
    # Exact increments must integrate to the supplied next state, including the off-centre object.
    for n in range(2):
        current, next_state = tuple(x[n:n+1] for x in S), tuple(x[n:n+1] for x in S1)
        with torch.no_grad():
            wm.net[-1].bias.copy_((wm.targets(current, next_state) / wm.out_norm.std())[0])
        predicted = wm.step(current, torch.zeros(1, 41), torch.zeros(1, 33))
        for a, b in zip(predicted, next_state, strict=True):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    yaw, shift = so3_exp(torch.tensor([0., 0., 1.2])), torch.tensor([2., -3., 0.])
    St, S1t = transform_state(S, yaw, shift), transform_state(S1, yaw, shift)
    torch.testing.assert_close(invariant_features(St), invariant_features(S), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(wm.targets(St, S1t), wm.targets(S, S1), atol=2e-6, rtol=1e-4)
    shifted = copy.copy(m)
    for key, value in zip(("root_p", "root_R", "root_v", "root_w", "full_q", "full_qd",
                           "object_p", "object_R", "object_v", "object_w"),
                          transform_state(m.ref_state(torch.arange(m.num_frames)), yaw, shift), strict=True):
        setattr(shifted, key, value)
    torch.testing.assert_close(shifted.goal(torch.tensor([90, 600]), 3), m.goal(torch.tensor([90, 600]), 3),
                               atol=2e-6, rtol=1e-4)


def test_observation_reward_and_critic_use_the_same_motion_phase(reference):
    m, mdp = reference
    f, a, xi = torch.tensor([120, 800]), torch.zeros(2, ACTION_DIM), torch.zeros(2, 33)
    S = m.ref_state(f)
    obs = mdp.obs(S, a, xi, f)
    assert obs.shape == (2, OBS_DIM) == (2, 186)
    torch.testing.assert_close(obs[:, 159:171], m.hand_reference(f))
    assert mdp.critic_features(S, f, xi).shape == (2, INV_DIM + CRITIC_EXTRA)
    _, perfect = mdp.reward(S, f, a, a, terms=True)
    torch.testing.assert_close(perfect["reward_object_pos"], torch.full((2,), 0.04))
    torch.testing.assert_close(perfect["reward_object_rot"], torch.full((2,), 0.04))
    # Actual root drift must not move the object's reference target.
    drift = (S[0] + torch.tensor([1., 0., 0.]), *S[1:])
    _, drift_terms = mdp.reward(drift, f, a, a, terms=True)
    torch.testing.assert_close(drift_terms["reward_object_pos"], perfect["reward_object_pos"])
    off = list(S)
    off[4] = (S[4] + 0.02).requires_grad_()
    off[6] = (S[6] + 0.03).requires_grad_()
    angle = torch.tensor([[0.1, 0.0, 0.0]]).expand(2, -1).clone().requires_grad_()
    off[7] = S[7] @ so3_exp(angle)
    reward, terms = mdp.reward(tuple(off), f, a, a, terms=True)
    torch.testing.assert_close(sum(terms[k] for k in REWARD_METRICS), reward)
    assert len(REWARD_METRICS) == 12  # Finger tracking removed; retain body, regularizers and two object terms.
    grads = torch.autograd.grad(reward.sum(), (off[4], off[6], angle))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_multistep_model_object_and_hand_action_gradients(reference):
    m, mdp = reference
    torch.manual_seed(2)
    wm = WorldModel(hidden=(16,))
    torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
    S = m.ref_state(torch.tensor([100, 700]))
    a = torch.randn(2, 41, requires_grad=True)
    for _ in range(3):
        S = wm.step(S, a, torch.zeros(2, 33))
    loss = mdp.reward(S, torch.tensor([103, 703]), a, torch.zeros_like(a)).sum()
    loss.backward()
    assert torch.isfinite(a.grad).all() and a.grad[:, 29:].abs().sum() > 0
    body, hand, obj = grouped_state_error(torch.ones(2, WM_OUT))
    torch.testing.assert_close(body, hand)
    torch.testing.assert_close(hand, obj)


def test_matching_parameter_initialization_keeps_new_normalizers_fresh():
    old = Policy(171, 146, hist=2, encoder_hidden=(16,), decoder_hidden=(16,))
    new = Policy(186, 224, hist=2, encoder_hidden=(16,), decoder_hidden=(16,))
    old.obs_norm.update(torch.randn(5, 171))
    report = initialize_matching_parameters(new, old.state_dict())
    assert "dec.2.weight" in report["copied"] and "dec.0.weight" in report["skipped"]
    assert new.obs_norm.count == 0 and new.goal_norm.count == 0


def test_recorded_object_and_full_reference_hand_render_in_world_frame(reference):
    import mujoco

    from supermimic.robot import ALL_HAND_JOINT_NAMES
    from supermimic.rotations import mat_to_quat
    from tools.render import TrajectoryPoseWriter, two_robot_xml

    motion, _ = reference
    f = 250
    model = mujoco.MjModel.from_xml_string(two_robot_xml(motion.object_path))
    data = mujoco.MjData(model)
    actual_object = motion.object_p[f].numpy() + [0.1, 0.2, 0]
    trajectory = {"root_p": motion.root_p[f:f+1, None].numpy(),
                  "root_quat": mat_to_quat(motion.root_R[f:f+1, None]).numpy(),
                  "q": motion.q[f:f+1, None].numpy(), "f": np.array([[f]]),
                  "hand_q": motion.hand_q[f:f+1, None].numpy() + 0.02,
                  "hand_joint_names": np.array(ALL_HAND_JOINT_NAMES),
                  "ref_root_p": motion.root_p.numpy(), "ref_root_R": motion.root_R.numpy(),
                  "ref_q": motion.q.numpy(), "ref_hand_q": motion.hand_q.numpy(),
                  "object_p": np.array([[actual_object]]),
                  "object_quat": mat_to_quat(motion.object_R[f:f+1, None]).numpy(),
                  "ref_object_p": motion.object_p.numpy(), "ref_object_R": motion.object_R.numpy()}
    writer = TrajectoryPoseWriter(model, trajectory)
    writer.write(data, 0, 0, global_ghost=True)
    np.testing.assert_allclose(data.qpos[writer.object["object/"][:3]], actual_object)
    np.testing.assert_allclose(data.qpos[writer.object["object_ref/"][:3]], motion.object_p[f])
    np.testing.assert_allclose(data.qpos[writer.hand["ref/"]], motion.hand_q[f])
    np.testing.assert_allclose(data.qpos[writer.hand["robot/"]], motion.hand_q[f] + 0.02)


@pytest.fixture(scope="module")
def gpu_env():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from supermimic.env import TrackingSim, make_sim

    device = "cuda:0"
    torch.cuda.set_device(device)
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(DEFAULT_MOTION_DIR, kin, device, clips=REFERENCE_CLIPS)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    return TrackingSim(make_sim(device), 4, motion, mdp, eval_mode=True)


def test_gpu_joint_reset_roundtrip_partial_reset_and_reward_phase(gpu_env):
    e = gpu_env
    ids = torch.arange(4, device=e.device)
    f = torch.tensor([40, 260, 650, 1100], device=e.device)
    e.reset(ids, f)
    for actual, expected in zip(e.state(), e.motion.ref_state(f), strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    out = e.step(torch.zeros(4, 41, device=e.device))
    torch.testing.assert_close(out["hand_reference"], e.motion.hand_reference(f + 1))
    torch.testing.assert_close(out["r"], e.mdp.reward(unflatten_state(out["s1"]), f + 1, out["a"], out["a_prev"]))
    before = e.state()
    e.reset(ids[::2], f[::2] + 10)
    for actual, expected, old in zip(e.state(), e.motion.ref_state(f + 10), before, strict=True):
        torch.testing.assert_close(actual[::2], expected[::2], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(actual[1::2], old[1::2], atol=0, rtol=0)


def test_gpu_object_is_dynamic_and_not_teleported_to_reference(gpu_env):
    from supermimic.rotations import mat_to_quat

    e = gpu_env
    p, R, _, _ = e._free_state(e.object)
    p[:, 2] += 1
    e.object.write_root_link_pose_to_sim(torch.cat((p, mat_to_quat(R)), -1))
    e.object.write_root_link_velocity_to_sim(torch.zeros(4, 6, device=e.device))
    e.refresh()
    for _ in range(5):
        e.step(torch.zeros(4, 41, device=e.device))
    state = e.state()
    assert (state[6][:, 2] < p[:, 2] - 0.02).all() and (state[8][:, 2] < -0.5).all()
    assert (state[6] - e.motion.object_p[e.f]).norm(dim=-1).min() > 0.5
    model = e.sim.mj_model
    robot_geoms, object_geoms = e.robot.indexing.geom_ids.cpu(), e.object.indexing.geom_ids.cpu()
    assert (model.geom_contype[robot_geoms] == 0).all()  # no self collision
    assert (model.geom_conaffinity[robot_geoms] == 5).any()
    assert (model.geom_contype[object_geoms] == 4).all()
    assert np.isclose(model.body_mass[e.object.indexing.root_body_id], 0.8762655984)


def test_gpu_robot_object_contacts_generate_free_joint_constraint_force(gpu_env):
    e = gpu_env
    frames = (e.motion.contact_label == 1).sum(-1).topk(e.N).indices
    e.reset(torch.arange(e.N, device=e.device), frames)
    out = e.step(torch.zeros(e.N, 41, device=e.device))
    assert (out["terms"]["object_penetration"] > 0).any()
    object_force = e.sim.data.qfrc_constraint[:, e.object.indexing.free_joint_v_adr]
    assert torch.isfinite(object_force).all() and object_force.abs().max() > 1e-3
