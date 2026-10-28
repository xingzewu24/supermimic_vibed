"""Reward contributions and termination diagnostics retain the MDP's behavior."""

import math
from types import SimpleNamespace

import pytest
import torch

from supermimic.config import HandControlConfig
from supermimic.kinematics import G1Kinematics
from supermimic.logger import REWARD_METRICS
from supermimic.mdp import ACTION_DIM, JOINT_DIM, MAX_STEP_REWARD, MDP, RewardCfg, body_state, tracking_errors
from supermimic.motion import CONTROL_DT
from supermimic.networks import WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import so3_exp


@pytest.fixture
def reference_mdp():
    batch = 4
    default_q = torch.tensor(default_joint_pos_list())
    p = torch.tensor([[0.0, 0.0, 0.8]]).expand(batch, -1).clone()
    R = torch.eye(3).expand(batch, -1, -1).clone()
    q = torch.zeros(batch, JOINT_DIM)
    q[:, :29] = default_q
    state = (p, R, torch.zeros(batch, 3), torch.zeros(batch, 3), q, torch.zeros_like(q),
             p + 0.5, R.clone(), torch.zeros(batch, 3), torch.zeros(batch, 3))
    kin = G1Kinematics("cpu")
    bp, bR, bv, bw = kin.forward(*body_state(state))
    motion = SimpleNamespace(root_p=p.clone(), root_R=R.clone(), q=q[:, :29].clone(),
                             body_p=bp, body_R=bR, body_v=bv, body_w=bw,
                             hand_reference=lambda f: q[f, 29:41], object_p=state[6], object_R=state[7],
                             object_v=state[8], object_w=state[9])
    mdp = MDP(
        kin, motion, default_q, RewardCfg(w_action_rate=-0.07, w_joint_limit=-1.5),
        hand_cfg=HandControlConfig(
            action_rate_weight=0.03, joint_limit_weight=3.0,
        ),
    )
    return mdp, state, torch.arange(batch)


def test_object_tracking_matches_reference_training_weights_and_shortest_angle(reference_mdp):
    mdp, state, frames = reference_mdp
    # Independent analytic cases, including rotations past pi and a nontrivial reference orientation.
    offsets = torch.tensor([[0., 0., 0.], [0.2, 0., 0.], [0.1, -0.2, 0.3], [-0.1, 0., 0.]])
    angles = torch.tensor([0., 0.3, math.pi, 2 * math.pi - 0.2])
    mdp.motion.object_R = so3_exp(torch.tensor([[0.2, -0.3, 0.1]]).expand(4, -1))
    axis = torch.tensor([1., 2., -3.])
    axis = axis / axis.norm()
    actual = list(state)
    actual[6] = mdp.motion.object_p + offsets
    actual[7] = mdp.motion.object_R @ so3_exp(angles[:, None] * axis)
    action = torch.zeros(4, ACTION_DIM)
    reward, terms = mdp.reward(tuple(actual), frames, action, action, terms=True)
    shortest_angles = torch.tensor([0., 0.3, math.pi, 0.2])
    expected_pos = 2 * torch.exp(-offsets.square().sum(-1) / 0.2**2) * CONTROL_DT
    expected_rot = 2 * torch.exp(-shortest_angles.square() / 0.3**2) * CONTROL_DT
    torch.testing.assert_close(terms["reward_object_pos"], expected_pos, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(terms["reward_object_rot"], expected_rot, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(terms["object_rotation_error"], shortest_angles, atol=1e-6, rtol=1e-5)
    assert mdp.max_step_reward == MAX_STEP_REWARD == 9 * CONTROL_DT
    assert (reward <= mdp.max_step_reward).all()


def test_object_rewards_have_finite_corrective_gradients(reference_mdp):
    mdp, state, frames = reference_mdp
    actual = list(state)
    offset = torch.tensor([[0.1, -0.05, 0.02]]).expand(4, -1).clone().requires_grad_()
    angle = torch.tensor([[0.1, -0.2, 0.05]]).expand(4, -1).clone().requires_grad_()
    actual[6] = mdp.motion.object_p + offset
    actual[7] = mdp.motion.object_R @ so3_exp(angle)
    action = torch.zeros(4, ACTION_DIM)
    _, terms = mdp.reward(tuple(actual), frames, action, action, terms=True)
    gradients = torch.autograd.grad((terms["reward_object_pos"] + terms["reward_object_rot"]).sum(),
                                    (offset, angle))
    for error, gradient in zip((offset, angle), gradients, strict=True):
        assert torch.isfinite(gradient).all()
        assert (error * gradient < 0).all()


def test_object_reward_alone_reaches_hand_actions_through_imagination(reference_mdp):
    mdp, state, frames = reference_mdp
    torch.manual_seed(17)
    wm = WorldModel(hidden=(16,))
    torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
    action = torch.zeros(4, ACTION_DIM, requires_grad=True)
    for _ in range(3):
        state = wm.step(state, action, torch.zeros(4, 33))
    _, terms = mdp.reward(state, frames, action, torch.zeros_like(action), terms=True)
    (terms["reward_object_pos"] + terms["reward_object_rot"]).sum().backward()
    assert torch.isfinite(action.grad).all()
    assert (action.grad[:, 29:].abs().sum(0) > 0).all()


def test_tracking_errors_report_mean_distances_and_angles_in_physical_units():
    q = torch.zeros(2, 29)
    q[0, 0] = 0.29
    q[1] = -0.3
    p = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    rotations = so3_exp(torch.tensor([
        [[math.pi / 2, 0.0, 0.0], [0.0, 0.0, math.pi / 6]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]))
    errors = tracking_errors(q, p, rotations, torch.zeros_like(q), torch.zeros_like(p),
                             torch.eye(3).expand(2, 2, -1, -1))
    torch.testing.assert_close(errors["joint_mae"], torch.tensor([0.01, 0.3]))
    torch.testing.assert_close(errors["body_position_error"], torch.tensor([3.5, 0.0]))
    torch.testing.assert_close(errors["body_rotation_error"], torch.tensor([math.pi / 3, 0.0]))


def test_reward_tracking_errors_reuse_fk_and_match_reference_frames(reference_mdp, monkeypatch):
    mdp, state, _ = reference_mdp
    mdp.motion.q += torch.arange(4)[:, None] * 0.1
    state = (state[0] + torch.tensor([0.2, 0.0, 0.0]), *state[1:])
    calls = []
    forward = mdp.kin.forward

    def counted_forward(*args):
        calls.append(True)
        return forward(*args)

    monkeypatch.setattr(mdp.kin, "forward", counted_forward)
    action = torch.zeros(4, ACTION_DIM)
    _, diagnostics = mdp.reward(state, torch.tensor([3, 2, 1, 0]), action, action, terms=True)
    assert len(calls) == 1
    torch.testing.assert_close(diagnostics["joint_mae"], torch.tensor([0.3, 0.2, 0.1, 0.0]))
    torch.testing.assert_close(diagnostics["body_position_error"], torch.full((4,), 0.2))
    torch.testing.assert_close(diagnostics["body_rotation_error"], torch.zeros(4), atol=1e-6, rtol=0)


def test_weighted_reward_contributions_sum_to_total_and_preserve_gradients(reference_mdp):
    mdp, state, frames = reference_mdp
    p, _, _, _, q, qd = state[:6]
    q = q.clone()
    q[:, 0] = mdp.soft_hi[0] + 0.2
    q[:, 29:] = mdp.hand.full_hi + 0.1
    q.requires_grad_()
    qd = qd.clone()
    qd[:, 29:] = 0.25
    rotations = so3_exp(torch.tensor([[0.1, 0.2, 0.3]]).expand(4, -1))
    state = (p + 0.05, rotations, torch.full((4, 3), 0.1), torch.full((4, 3), 0.2), q, qd, *state[6:])
    action = torch.full((4, ACTION_DIM), 0.4, requires_grad=True)
    previous = torch.zeros_like(action)
    plain_reward = mdp.reward(state, frames, action, previous)
    reward, diagnostics = mdp.reward(state, frames, action, previous, terms=True)
    torch.testing.assert_close(reward, plain_reward)
    torch.testing.assert_close(sum(diagnostics[name] for name in REWARD_METRICS), reward)
    for name in REWARD_METRICS:
        if name in ("reward_action_rate", "reward_joint_limit", "reward_hand_action_rate", "reward_hand_joint_limit"):
            assert (diagnostics[name] < 0).all()
        else:
            assert (diagnostics[name] >= 0).all()
    expected_gradients = torch.autograd.grad(plain_reward.sum(), (q, action))
    actual_gradients = torch.autograd.grad(reward.sum(), (q, action))
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_hand_speed_does_not_change_reward_even_with_a_legacy_weight(reference_mdp):
    mdp, state, frames = reference_mdp
    action = torch.zeros(4, ACTION_DIM)
    expected = mdp.reward(state, frames, action, action)
    # Old checkpoint parameters must not reactivate the retired penalty.
    mdp.hand.cfg.velocity_weight = 1000.0
    velocity = state[5].clone()
    velocity[:, 29:41] = 20.0
    velocity[:, 41:] = -30.0
    velocity.requires_grad_()
    moving = (*state[:5], velocity, *state[6:])
    reward, diagnostics = mdp.reward(moving, frames, action, action, terms=True)
    torch.testing.assert_close(reward, expected, atol=0, rtol=0)
    gradient = torch.autograd.grad(reward.sum(), velocity)[0]
    torch.testing.assert_close(gradient[:, 29:], torch.zeros_like(gradient[:, 29:]), atol=0, rtol=0)
    assert "reward_hand_velocity" not in diagnostics
    assert "reward_hand_velocity" not in REWARD_METRICS


def test_termination_diagnostics_include_simultaneous_height_and_tilt(reference_mdp):
    mdp, state, frames = reference_mdp
    p = state[0].clone()
    p[:, 2] += torch.tensor([0.0, 0.3, 0.0, 0.3])
    R = so3_exp(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                             [math.pi / 2, 0.0, 0.0], [math.pi / 2, 0.0, 0.0]]))
    state = (p, R, *state[2:])
    terminated, diagnostics = mdp.termination(state, frames, terms=True)
    torch.testing.assert_close(terminated, torch.tensor([False, True, True, True]))
    torch.testing.assert_close(terminated, mdp.termination(state, frames))
    torch.testing.assert_close(diagnostics["height"], torch.tensor([False, True, False, True]))
    torch.testing.assert_close(diagnostics["tilt"], torch.tensor([False, False, True, True]))
    torch.testing.assert_close(diagnostics["height_error"], torch.tensor([0.0, 0.3, 0.0, 0.3]))
    torch.testing.assert_close(diagnostics["tilt_error"], torch.tensor([0.0, 0.0, 1.0, 1.0]))
