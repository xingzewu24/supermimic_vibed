"""Full hand FK against independent MuJoCo poses, velocities and finite differences."""

import mujoco
import numpy as np
import pytest
import torch

from supermimic.kinematics import G1Kinematics
from supermimic.mdp import ACTION_DIM
from supermimic.networks import WorldModel
from supermimic.robot import ALL_HAND_JOINT_NAMES, G1_JOINT_NAMES, TRACKED_BODIES, g1_spec
from supermimic.rotations import mat_to_quat, so3_exp


@pytest.fixture(scope="module")
def kin():
    return G1Kinematics("cpu")


def state(batch=3):
    torch.manual_seed(108)
    # Independently sampled followers deliberately violate ideal mimic constraints.
    return (torch.randn(batch, 3), so3_exp(torch.randn(batch, 3)),
            torch.randn(batch, 3), torch.randn(batch, 3),
            torch.rand(batch, 53) * 0.8, torch.randn(batch, 53))


def test_all_links_and_collision_points_match_mujoco(kin):
    model = g1_spec().compile()
    data = mujoco.MjData(model)
    S = state()
    poses = kin.forward(*S, all_bodies=True)
    surfaces = kin.hand_surface(*S)
    assert len(kin.hand_body_names) == 26
    assert len(kin.hand_surface_geom_ids) > 100
    sampled = {model.body(int(model.geom_bodyid[g])).name for g in kin.hand_surface_geom_ids}
    assert sampled == set(kin.hand_body_names)
    assert all(model.geom_conaffinity[g] for g in kin.hand_surface_geom_ids)
    assert set(kin.joint_names) == set(G1_JOINT_NAMES + ALL_HAND_JOINT_NAMES)
    for i in range(len(S[0])):
        p, R, v, w, q, qd = (x[i] for x in S)
        data.qpos[:3], data.qpos[3:7] = p.numpy(), mat_to_quat(R).numpy()
        data.qvel[:3], data.qvel[3:6] = v.numpy(), (R.T @ w).numpy()
        for j, name in enumerate(kin.joint_names):
            joint = model.joint(name)
            data.qpos[joint.qposadr[0]], data.qvel[joint.dofadr[0]] = q[j], qd[j]
        mujoco.mj_forward(model, data)
        origin_velocities = {}
        for b, name in enumerate(kin.body_names):
            bid = model.body(name).id
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 0)
            origin_v = velocity[3:] - np.cross(velocity[:3], data.xipos[bid] - data.xpos[bid])
            origin_velocities[name] = (origin_v, velocity[:3].copy())
            expected = (data.xpos[bid], data.xmat[bid].reshape(3, 3), origin_v, velocity[:3])
            for actual, target in zip(poses, expected, strict=True):
                np.testing.assert_allclose(actual[i, b], target, atol=4e-6, rtol=1e-5)
        for k, g in enumerate(kin.hand_surface_geom_ids):
            bid = int(model.geom_bodyid[g])
            name = model.body(bid).name
            offset = data.xmat[bid].reshape(3, 3) @ kin.hand_surface_local[k].numpy()
            expected_p = data.xpos[bid] + offset
            origin_v, angular_v = origin_velocities[name]
            expected_v = origin_v + np.cross(angular_v, offset)
            np.testing.assert_allclose(surfaces[0][i, k], expected_p, atol=4e-6, rtol=1e-5)
            np.testing.assert_allclose(surfaces[1][i, k], expected_v, atol=5e-6, rtol=1e-5)
            # Samples must be on the collision mesh, including its geom transform.
            mesh = int(model.geom_dataid[g])
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world_vertices = (model.mesh_vert[start:start + count] @ data.geom_xmat[g].reshape(3, 3).T
                              + data.geom_xpos[g])
            assert np.linalg.norm(world_vertices - expected_p, axis=-1).min() < 1e-7


def test_body_api_and_arbitrary_batch_dimensions(kin):
    S = state(6)
    full = kin.forward(*S)
    body_only = kin.forward(*S[:4], S[4][:, :29], S[5][:, :29])
    batched = kin.forward(*(x.reshape(2, 3, *x.shape[1:]) for x in S))
    for actual, expected, nested in zip(full, body_only, batched, strict=True):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, nested.flatten(0, 1))
    assert full[0].shape == (6, len(TRACKED_BODIES), 3)
    for forward in (kin.forward_hands, kin.hand_surface):
        with pytest.raises(ValueError, match="actual followers"):
            forward(*S[:4], S[4][:, :29], S[5][:, :29])
    with pytest.raises(ValueError, match="actual followers"):
        kin.forward(*S[:4], S[4][:, :29], S[5][:, :29], all_bodies=True)


def test_each_actual_finger_has_geometry_and_velocity_gradients(kin):
    S = list(state(1))
    S[4].requires_grad_()
    S[5].requires_grad_()
    points, velocities = kin.hand_surface(*S)
    weights = torch.randn_like(points)
    loss = (points * weights).sum() + (velocities * weights).sum() * 0.1
    q_grad, qd_grad = torch.autograd.grad(loss, S[4:6])
    assert torch.isfinite(q_grad).all() and torch.isfinite(qd_grad).all()
    assert (q_grad[:, 29:].abs() > 1e-6).all()
    assert (qd_grad[:, 29:].abs() > 1e-6).all()
    # Every follower is independent: compare geometric derivatives to finite differences.
    for j in range(41, 53):
        shifted = [x.detach().clone() for x in S]
        shifted[4][0, j] += 1e-3
        plus = kin.hand_surface(*shifted)
        shifted[4][0, j] -= 2e-3
        minus = kin.hand_surface(*shifted)
        derivative = (((plus[0] - minus[0]) * weights).sum()
                      + ((plus[1] - minus[1]) * weights).sum() * 0.1) / 2e-3
        torch.testing.assert_close(q_grad[0, j], derivative, atol=5e-4, rtol=3e-3)


def test_imagination_gradients_reach_hand_actions_through_surface_fk(kin):
    wm = WorldModel(hidden=(16,))
    torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
    S = (*state(2), torch.ones(2, 3), torch.eye(3).expand(2, -1, -1),
         torch.zeros(2, 3), torch.zeros(2, 3))
    action = torch.randn(2, ACTION_DIM, requires_grad=True)
    for _ in range(3):
        S = wm.step(S, action, torch.zeros(2, 33))
    points, velocities = kin.hand_surface(*S[:6])
    (points.square().mean() + velocities.square().mean()).backward()
    assert torch.isfinite(action.grad).all()
    assert (action.grad[:, 29:].abs().sum(0) > 0).all()
