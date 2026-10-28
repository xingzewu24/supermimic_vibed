"""Hand actuator mapping, joint dynamics, and replay phase reconstruction."""

from types import SimpleNamespace

import torch

from supermimic.buffer import FIELDS, ReplayBuffer
from supermimic.hand import HandControl
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import (
    ACTION_DIM,
    CRITIC_EXTRA,
    INV_DIM,
    JOINT_DIM,
    MDP,
    OBS_DIM,
    STATE_DIM,
    body_state,
    flatten_state,
    unflatten_state,
)
from supermimic.networks import WM_OUT, Policy, WorldModel
from supermimic.robot import default_joint_pos_list, hand_joint_positions
from supermimic.rotations import so3_exp


def full_state(batch=2):
    return (torch.zeros(batch, 3), torch.eye(3).expand(batch, -1, -1).clone(),
            torch.zeros(batch, 3), torch.zeros(batch, 3), torch.zeros(batch, JOINT_DIM),
            torch.zeros(batch, JOINT_DIM), torch.ones(batch, 3),
            torch.eye(3).expand(batch, -1, -1).clone(), torch.zeros(batch, 3), torch.zeros(batch, 3))


def test_driver_ranges_respect_both_assets_and_all_followers():
    hand = HandControl("cpu")
    # Interior deterministic actions avoid float32 endpoints rounding outside strict Python limits.
    for value in (hand.lo + 1e-6, hand.hi - 1e-6, hand.action_target(torch.linspace(-3, 3, 12))):
        expanded = hand_joint_positions(value.tolist())
        torch.testing.assert_close(hand.expand(value), torch.tensor(list(expanded.values())))
    assert hand.hi[1] < 0.6  # thumb pitch is limited by its distal follower


def test_action_targets_saturate_without_changing_mimic_coordinates():
    hand = HandControl("cpu")
    torch.testing.assert_close(hand.action_target(torch.full((12,), -100.0)), hand.lo)
    torch.testing.assert_close(hand.action_target(torch.full((12,), 100.0)), hand.hi)
    value = torch.zeros(12, requires_grad=True)
    hand.expand(hand.action_target(value)).sum().backward()
    assert torch.isfinite(value.grad).all() and (value.grad > 0).all()


def test_world_model_full_state_roundtrip_and_rotation_increment():
    torch.manual_seed(3)
    S, S1 = full_state(1), full_state(1)
    S = tuple(x + torch.randn_like(x) * 0.01 if i not in (1, 7) else so3_exp(torch.randn(1, 3) * 0.1)
              for i, x in enumerate(S))
    S1 = tuple(x + torch.randn_like(x) * 0.03 if i not in (1, 7) else x @ so3_exp(torch.randn(1, 3) * 0.1)
               for i, x in enumerate(S))
    packed = flatten_state(S)
    assert packed.shape == (1, STATE_DIM) == (1, 132)
    for actual, expected in zip(unflatten_state(packed), S):
        torch.testing.assert_close(actual, expected)
    wm = WorldModel(hidden=(16,))
    assert wm.in_dim == 208 and WM_OUT == 130
    with torch.no_grad():
        wm.net[-1].bias.copy_((wm.targets(S, S1) / wm.out_norm.std())[0])
    predicted = wm.step(S, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33))
    for actual, expected in zip(predicted, S1):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_hand_actions_and_states_have_gradients_through_multiple_model_steps():
    torch.manual_seed(9)
    wm = WorldModel(hidden=(16,))
    torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
    S = tuple(x.requires_grad_() for x in full_state())
    action = torch.randn(2, ACTION_DIM, requires_grad=True)
    pred = S
    for _ in range(3):
        pred = wm.step(pred, action, torch.zeros(2, 33))
    loss = pred[4][..., :29].square().sum() + pred[4][..., 29:].square().sum()
    loss.backward()
    assert torch.isfinite(action.grad).all() and action.grad[:, 29:].abs().sum() > 0
    assert torch.isfinite(S[4].grad).all() and S[4].grad[:, 29:].abs().sum() > 0


def test_motion_hand_reference_enters_policy_and_critic_without_tracking_diagnostics():
    hand = HandControl("cpu")
    S = list(full_state())
    S[0][:, 2] = 0.8
    S[4][:, :29] = torch.tensor(default_joint_pos_list())
    S[4][:, 29:] = hand.expand(hand.mid)
    S = tuple(S)
    kin = G1Kinematics("cpu")
    bp, bR, bv, bw = kin.forward(*body_state(S))
    motion = SimpleNamespace(root_p=S[0], root_R=S[1], q=S[4][:, :29],
                             body_p=bp, body_R=bR, body_v=bv, body_w=bw,
                             hand_q=S[4][:, 29:41].clone(), object_p=S[6], object_R=S[7],
                             object_v=S[8], object_w=S[9])
    motion.hand_reference = lambda f: motion.hand_q[f]
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
    f, xi, a = torch.arange(2), torch.zeros(2, 33), torch.zeros(2, ACTION_DIM)
    r = mdp.reward(S, f, a, a)
    motion.hand_q += 0.1
    reward, diagnostics = mdp.reward(S, f, a, a, terms=True)
    torch.testing.assert_close(r, reward, atol=0, rtol=0)
    assert not {"hand_track", "e_hand", "reward_hand_track", "hand_mae", "hand_mimic_error",
                "hand_action_rate"} & diagnostics.keys()
    shifted = list(S)
    shifted[4] = S[4].clone().requires_grad_()
    mdp.reward(tuple(shifted), f, a, a).sum().backward()
    torch.testing.assert_close(shifted[4].grad[:, 29:], torch.zeros(2, 24), atol=0, rtol=0)
    obs = mdp.obs(S, a, xi, f)
    assert obs.shape == (2, OBS_DIM) == (2, 186)
    torch.testing.assert_close(obs[:, 159:171], motion.hand_reference(f))
    features = mdp.critic_features(S, f, xi)
    assert features.shape == (2, INV_DIM + CRITIC_EXTRA)
    pi = Policy(OBS_DIM, 146, hist=2, encoder_hidden=(16,), decoder_hidden=(16,))
    assert pi(obs[:, None].expand(-1, 2, -1), torch.zeros(2, 146)).shape == (2, 41)


def test_replay_retains_motion_phase_and_masks_resets():
    replay = ReplayBuffer(8, 2, "cpu")
    for t in range(8):
        tr = {k: torch.zeros((2,) + ((shape,) if isinstance(shape, int) else shape), dtype=dtype)
              for k, (shape, dtype) in FIELDS.items()}
        tr["ep_step"][:] = t % 4
        tr["f"][:] = (0 if t < 4 else 20) + t % 4
        tr["done"][:] = t % 4 == 3
        replay.add(tr)
    starts, hist = replay.sample_starts(64, 3)
    assert starts["s"].shape[-1] == 132 and starts["a_prev"].shape[-1] == 41
    assert (hist["f"] - hist["ep_step"] == (starts["f"] - starts["ep_step"])[:, None]).all()
    assert "hand_seed" not in replay.data and "hand_target" not in replay.data
    windows = replay.sample_windows(64, 4)
    expected = torch.cumprod(torch.cat((torch.ones(64, 1), 1 - windows["done"][:, :-1].float()), -1), -1)
    torch.testing.assert_close(windows["mask"], expected)
