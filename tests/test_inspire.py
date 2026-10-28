"""Full-asset topology, physical hand coupling, body FK and trajectory restoration."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.env import make_scene
from supermimic.kinematics import G1Kinematics
from supermimic.robot import (
    ALL_HAND_JOINT_NAMES,
    G1_JOINT_NAMES,
    HAND_ARMATURE,
    HAND_DAMPING,
    HAND_JOINT_NAMES,
    HAND_STIFFNESS,
    TRACKED_BODIES,
    URDF_PATH,
    _kp,
    action_scale,
    default_joint_pos_list,
    g1_spec,
    hand_joint_positions,
    mimic_relations,
    motor_of,
)
from supermimic.rotations import mat_to_quat, so3_exp

POSE = [0.3, 0.2, 0.4, 0.5, 0.6, 0.7] * 2


@pytest.fixture(scope="module")
def model():
    return make_scene(1, "cpu", True).compile()


def test_full_model_and_attached_mimic_references(model):
    assert (model.nq, model.nv, model.nu, model.neq) == (60, 59, 53, 12)
    joints = {model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
    assert joints == {f"robot/{n}" for n in G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES}
    driven = {model.joint(int(i)).name for i in model.actuator_trnid[:, 0]}
    assert driven == {f"robot/{n}" for n in G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES}
    for e, (follower, (driver, multiplier, offset)) in enumerate(mimic_relations().items()):
        assert model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT
        assert model.joint(int(model.eq_obj1id[e])).name == f"robot/{follower}"
        assert model.joint(int(model.eq_obj2id[e])).name == f"robot/{driver}"
        np.testing.assert_allclose(model.eq_data[e, :5], [offset, multiplier, 0, 0, 0])
    for name in G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES:
        j = model.joint(f"robot/{name}").id
        a = np.flatnonzero(model.actuator_trnid[:, 0] == j).item()
        effort = motor_of(name)[0] if name in G1_JOINT_NAMES else 1.0
        np.testing.assert_allclose(model.jnt_actfrcrange[j], [-effort, effort])
        np.testing.assert_allclose(model.actuator_forcerange[a], [-effort, effort])
        if name in G1_JOINT_NAMES:
            assert action_scale(name) == pytest.approx(0.25 * effort / _kp(motor_of(name)[2]))
        else:
            assert model.actuator_gainprm[a, 0] == HAND_STIFFNESS == 5.0
            assert model.actuator_biasprm[a, 2] == -HAND_DAMPING == -0.5
            assert model.dof_armature[model.jnt_dofadr[j]] == HAND_ARMATURE == 0.01


@pytest.mark.parametrize("target,match", [
    ([0] * 11, "12 finite"), ([0] * 13, "12 finite"), ([float("nan")] * 12, "12 finite"),
    ([float("inf")] * 12, "12 finite"), ([2.0] + [0] * 11, "L_thumb_proximal_yaw"),
    ([0, 0.6] + [0] * 10, "L_thumb_intermediate"),
    ([0, -0.05] + [0] * 10, "L_thumb_intermediate"),
])
def test_hand_target_validates_drivers_and_followers(target, match):
    with pytest.raises(ValueError, match=match):
        hand_joint_positions(target)


def test_static_mimic_pose_expansion():
    positions = hand_joint_positions(POSE)
    assert positions["L_thumb_intermediate_joint"] == pytest.approx(0.32)
    assert positions["R_thumb_distal_joint"] == pytest.approx(0.48)


def test_urdf_offset_and_multiplier_are_preserved(monkeypatch):
    from supermimic import robot

    relations = mimic_relations()
    relations["L_index_intermediate_joint"] = ("L_index_proximal_joint", 0.7, 0.03)
    monkeypatch.setattr(robot, "mimic_relations", lambda: relations)
    m = g1_spec().compile()
    e = m.equality("mimic_L_index_intermediate_joint").id
    np.testing.assert_allclose(m.eq_data[e, :5], [0.03, 0.7, 0, 0, 0])
    assert hand_joint_positions(POSE)["L_index_intermediate_joint"] == pytest.approx(0.31)


def test_nonzero_pose_holds_and_mimic_recovers_under_external_torque():
    # Remove gravity/contact so this test isolates actuator and equality dynamics.
    m = make_scene(1, "cpu", True).compile()
    m.opt.gravity[:] = 0
    m.opt.timestep = 0.005
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    d = mujoco.MjData(m)
    d.qpos[2] = 2.0
    targets = dict(zip(G1_JOINT_NAMES, default_joint_pos_list())) | hand_joint_positions(POSE)
    for name, value in targets.items():
        d.qpos[m.joint(f"robot/{name}").qposadr[0]] = value
    for a in range(m.nu):
        d.ctrl[a] = targets[m.joint(int(m.actuator_trnid[a, 0])).name.removeprefix("robot/")]
    follower = m.joint("robot/L_thumb_distal_joint")
    d.qpos[follower.qposadr[0]] += 0.15
    mujoco.mj_step(m, d, nstep=250)
    d.qfrc_applied[follower.dofadr[0]] = 0.02
    mujoco.mj_step(m, d, nstep=100)
    for name, (driver, multiplier, offset) in mimic_relations().items():
        q1 = d.qpos[m.joint(f"robot/{name}").qposadr[0]]
        q2 = d.qpos[m.joint(f"robot/{driver}").qposadr[0]]
        assert abs(q1 - multiplier * q2 - offset) < 0.002
    under_load = d.qpos[m.joint("robot/L_thumb_proximal_pitch_joint").qposadr[0]]
    assert abs(under_load - POSE[1]) > 0.001  # physical compliance, not kinematic locking
    d.qfrc_applied[:] = 0
    mujoco.mj_step(m, d, nstep=250)
    for name in HAND_JOINT_NAMES:
        assert abs(d.qpos[m.joint(f"robot/{name}").qposadr[0]] - targets[name]) < 0.002


def test_body_fk_poses_velocities_and_gradients_match_full_mujoco():
    torch.manual_seed(47)
    kin = G1Kinematics("cpu")
    assert not set(kin.body_names) & set(ALL_HAND_JOINT_NAMES)
    assert kin.num_bodies == 54
    assert kin.cache_key()["urdf_sha256"] == hashlib.sha256(Path(URDF_PATH).read_bytes()).hexdigest()
    assert kin.cache_key()["fk_version"] == 3
    m = g1_spec().compile()
    d = mujoco.MjData(m)
    B = 20
    p = torch.randn(B, 3)
    R = so3_exp(torch.randn(B, 3))
    v, w = torch.randn(B, 3), torch.randn(B, 3)
    q = (kin.q_lo + torch.rand(B, 29) * (kin.q_hi - kin.q_lo)).requires_grad_()
    qd = torch.randn(B, 29)
    bp, bR, bv, bw = kin.forward(p, R, v, w, q, qd)
    for i in range(B):
        d.qpos[:3] = p[i].numpy()
        d.qpos[3:7] = mat_to_quat(R[i]).numpy()
        d.qvel[:3] = v[i].numpy()
        d.qvel[3:6] = (R[i].T @ w[i]).numpy()
        for j, name in enumerate(G1_JOINT_NAMES):
            joint = m.joint(name)
            d.qpos[joint.qposadr[0]] = q[i, j].detach().numpy()
            d.qvel[joint.dofadr[0]] = qd[i, j].numpy()
        for name, value in hand_joint_positions(POSE).items():
            d.qpos[m.joint(name).qposadr[0]] = value
        mujoco.mj_forward(m, d)
        for b, name in enumerate(TRACKED_BODIES):
            bid = m.body(name).id
            np.testing.assert_allclose(bp[i, b].detach(), d.xpos[bid], atol=2e-6)
            np.testing.assert_allclose(bR[i, b].detach(), d.xmat[bid].reshape(3, 3), atol=2e-6)
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 0)
            origin_v = velocity[3:] - np.cross(velocity[:3], d.xipos[bid] - d.xpos[bid])
            np.testing.assert_allclose(bv[i, b].detach(), origin_v, atol=4e-6)
            np.testing.assert_allclose(bw[i, b].detach(), velocity[:3], atol=4e-6)
    bp.square().sum().backward()
    assert q.grad.shape == (B, 29) and torch.isfinite(q.grad).all()


def test_offline_render_restores_named_body_and_actual_finger_coordinates():
    from tools.render import TrajectoryPoseWriter, two_robot_xml

    m = mujoco.MjModel.from_xml_string(two_robot_xml())
    d = mujoco.MjData(m)
    assert (m.nq, m.nv, m.neq) == (120, 118, 24)
    for g in range(m.ngeom):
        if m.body(int(m.geom_bodyid[g])).name.startswith("ref/"):
            assert m.geom_rgba[g, 3] == pytest.approx(0.25 if m.geom_group[g] == 1 else 0.0)
    # Reverse recorded finger order to ensure rendering consumes names.
    names = list(reversed(ALL_HAND_JOINT_NAMES))
    actual = np.arange(24) * 0.01
    trajectory = {
        "root_p": np.array([[[1.0, 2.0, 0.8]]]), "root_quat": np.array([[[1, 0, 0, 0]]]),
        "q": np.arange(29).reshape(1, 1, 29) * 0.01, "f": np.array([[0]]),
        "ref_root_p": np.array([[3.0, 4.0, 0.85]]), "ref_root_R": np.eye(3)[None],
        "ref_q": np.arange(29)[None] * -0.01, "hand_target": np.array(POSE),
        "hand_joint_names": np.array(names), "hand_q": actual.reshape(1, 1, 24),
    }
    writer = TrajectoryPoseWriter(m, trajectory)
    writer.write(d, 0, 0)
    for i, name in enumerate(G1_JOINT_NAMES):
        assert d.qpos[m.joint("robot/" + name).qposadr[0]] == pytest.approx(i * 0.01)
        assert d.qpos[m.joint("ref/" + name).qposadr[0]] == pytest.approx(i * -0.01)
    for name in ALL_HAND_JOINT_NAMES:
        assert d.qpos[m.joint("robot/" + name).qposadr[0]] == pytest.approx(actual[names.index(name)])
        assert d.qpos[m.joint("ref/" + name).qposadr[0]] == pytest.approx(hand_joint_positions(POSE)[name])
    np.testing.assert_allclose(d.qpos[writer.root["ref/"][:3]], [1, 2, 0.85])
    writer.write(d, 0, 0, global_ghost=True)
    np.testing.assert_allclose(d.qpos[writer.root["ref/"][:3]], [3, 4, 0.85])
    # A per-frame command drives only the reference hand; actual recorded fingers remain intact.
    trajectory["hand_command"] = np.zeros((1, 1, 12))
    writer.write(d, 0, 0)
    np.testing.assert_allclose(d.qpos[writer.hand["ref/"]], 0)
    np.testing.assert_allclose(d.qpos[writer.hand["robot/"]], actual[writer.recorded_hand])
