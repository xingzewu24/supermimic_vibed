"""Checkpoint playback, clip restarts and real headless Viser scene checks."""

import socket
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from supermimic.checkpoint import load_policy
from supermimic.config import REPO, PlayConfig, TrainConfig
from supermimic.mdp import OBS_DIM
from supermimic.networks import Policy
from supermimic.play import create_player, sync_scene
from supermimic.robot import MJCF_PATH


@pytest.fixture
def checkpoint(tmp_path):
    cfg = OmegaConf.structured(TrainConfig)
    cfg.hist = 3
    cfg.goal_frames = 2
    cfg.amp = "none"
    cfg.network.encoder_hidden = [16]
    cfg.network.decoder_hidden = [16]
    cfg.network.latent_dim = 4
    policy = Policy(
        OBS_DIM, 112 * cfg.goal_frames, hist=cfg.hist, z_dim=cfg.network.latent_dim,
        encoder_hidden=cfg.network.encoder_hidden, decoder_hidden=cfg.network.decoder_hidden,
    )
    return {"args": OmegaConf.to_container(cfg, resolve=True), "pi": policy.state_dict()}


def test_checkpoint_restores_policy_architecture_weights_and_eval_mode(checkpoint):
    policy = load_policy(checkpoint, "cpu")
    assert not policy.training
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint["pi"][name])
    with torch.no_grad():
        action = policy(torch.zeros(1, 3, OBS_DIM), torch.zeros(1, 224))
    assert action.shape == (1, 41) and torch.isfinite(action).all()


def test_body_only_checkpoint_is_rejected_before_loading_weights(checkpoint):
    del checkpoint["args"]["control_version"]
    with pytest.raises(ValueError, match="HOI checkpoint"):
        load_policy(checkpoint, "cpu")


@pytest.fixture
def player(tmp_path, checkpoint):
    if not torch.cuda.is_available() or not Path(MJCF_PATH).is_file():
        pytest.skip("requires a CUDA GPU and downloaded G1 model")
    checkpoint["args"]["motion_dir"] = str(REPO / "data/hoi/motions/test")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(checkpoint, run_dir / "last.pt")
    cfg = OmegaConf.structured(PlayConfig)
    cfg.run_dir = str(run_dir)
    cfg.motion_dir = checkpoint["args"]["motion_dir"]
    return create_player(cfg)


def test_clip_selection_history_and_terminal_restart(player):
    env = player.env
    assert len(env.motion.names) >= 2
    assert player.clip == env.motion.names[0]
    assert int(env.f[0]) == int(env.motion.clip_start[0])
    torch.testing.assert_close(env.state()[4][:, 29:], env.motion.hand_q[env.f])
    player.step()
    assert player.steps == 1 and player.history.shape == (1, 3, OBS_DIM)
    assert torch.isfinite(player.history).all()

    player.reset(env.motion.names[1])
    assert player.history is None and player.steps == 0
    assert not player.finished and not player.failed
    assert not env.done_eval.any() and not env.failed_eval.any()
    assert int(env.f[0]) == int(env.motion.clip_start[1])
    env.reset(torch.zeros(1, dtype=torch.long, device=env.device), env.motion.clip_end[1:2] - 1)
    player.step()
    assert player.finished
    before = env.sim.data.time.clone()
    player.step()
    torch.testing.assert_close(env.sim.data.time.clone(), before)

    player.reset(env.motion.names[0])
    assert not player.finished and not env.done_eval.any() and not env.failed_eval.any()
    player.step()
    assert player.steps == 1
    with pytest.raises(ValueError, match="Unknown clip"):
        player.reset("not_a_motion")


def test_seek_replays_cached_frames_and_preserves_policy_history(player):
    assert player.seek(4)
    assert player.steps == player.cached_steps == 4
    expected_return = player.return_sum
    expected_qpos = player.current_frame.qpos.copy()
    physics_time = player.env.sim.data.time.clone()
    history = player.history.clone()
    reference_frame = int(player.env.f[0])

    assert player.seek(1)
    assert player.steps == 1 and player.cached_steps == 4
    assert player.current_frame.reference_frame == reference_frame - 3
    assert player.return_sum == player.current_frame.return_sum
    assert not player.finished and not player.failed
    player.step()
    assert player.steps == 2
    torch.testing.assert_close(player.env.sim.data.time.clone(), physics_time)
    torch.testing.assert_close(player.history, history)
    np.testing.assert_array_equal(player.env.sim.data.qpos[0].cpu(), expected_qpos)

    player.seek(4)
    assert player.return_sum == expected_return
    env = player.env
    observation = env.mdp.obs(env.state(), env.a_prev, env.xi, env.f)
    expected_history = torch.cat((history[:, 1:], observation.unsqueeze(1)), 1)
    player.step()
    assert player.steps == player.cached_steps == 5
    assert int(player.env.f[0]) == reference_frame + 1
    np.testing.assert_array_equal(player._frames[4].qpos, expected_qpos)
    torch.testing.assert_close(player.history, expected_history, atol=0, rtol=0)


def test_seek_budget_and_early_termination(player):
    assert not player.seek(5, max_steps=2)
    assert player.steps == 2
    player.seek(0)
    assert not player.seek(5, max_steps=2)
    assert player.steps == 4
    assert player.seek(5, max_steps=2)
    for frame in (-1, player.last_frame + 1):
        with pytest.raises(ValueError, match="Frame must be"):
            player.seek(frame)
    with pytest.raises(ValueError, match="max_steps must be positive"):
        player.seek(0, max_steps=0)

    player.reset(player.clip)
    player.env.max_episode_steps = 2
    assert player.seek(player.last_frame, max_steps=8)
    assert player.finished and player.steps == player.cached_steps == 2
    physics_time = player.env.sim.data.time.clone()
    player.seek(0)
    assert not player.finished and player.return_sum == 0.0
    player.step()
    assert not player.finished
    player.step()
    assert player.finished and player.steps == 2
    player.step()
    torch.testing.assert_close(player.env.sim.data.time.clone(), physics_time)
    player.reset(player.env.motion.names[1])
    assert player.steps == player.cached_steps == 0
    assert player.current_frame.reference_frame == int(player.env.motion.clip_start[1])
    assert not player.finished and not player.failed


def test_playback_controls_seek_pause_and_switch_clips(player, monkeypatch):
    import viser

    from supermimic import play

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
    monkeypatch.setattr(play, "create_player", lambda cfg: player)
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    monkeypatch.setattr(server, "get_clients", lambda: {1: object()})
    cfg = OmegaConf.structured(PlayConfig)
    cfg.paused = True
    clock, tick = 0.0, 0

    def controls():
        return {handle.label: handle for handle in server.gui._gui_input_handle_from_uuid.values()}

    def trigger(label, value=None):
        handle = controls()[label]
        if value is not None:
            handle.value = value
        event = viser.GuiEvent(client_id=1, client=None, target=handle)
        for callback in handle._impl.update_cb:
            callback(event)

    def advance(_duration):
        nonlocal clock, tick
        gui = controls()
        if tick == 0:
            assert player.steps == 0
            trigger("Frame", 3)
        elif tick == 1:
            assert player.steps == gui["Frame"].value == 3
            assert gui["Paused"].value
            trigger("Previous frame")
        elif tick == 2:
            assert player.steps == 2 and player.cached_steps == 3
            trigger("Single step")
        elif tick == 3:
            assert player.steps == 3
            trigger("Forward 5 s")
        elif tick == 4:
            assert 3 < player.steps <= 11
            trigger("Back 5 s")
        elif tick == 5:
            assert player.steps == gui["Frame"].value == 0
            gui["Paused"].value = False
        elif tick == 6:
            assert player.steps == 1 and not gui["Paused"].value
            gui["Paused"].value = True
            gui["Motion"].value = player.env.motion.names[1]
        elif tick == 7:
            assert player.clip == player.env.motion.names[1]
            assert player.steps == player.cached_steps == gui["Frame"].value == 0
            assert gui["Frame"].max == player.last_frame
            raise KeyboardInterrupt
        clock += 0.03
        tick += 1

    # Drive actual Viser callbacks and the viewer loop without a browser or sleeps.
    monkeypatch.setattr(play, "time", SimpleNamespace(monotonic=lambda: clock, sleep=advance))
    play.play_run(cfg)
    assert tick == 7


def test_headless_viser_serves_scene_and_displays_integrated_pose(player):
    import mujoco
    import viser
    from mjviser import ViserMujocoScene

    from supermimic.ghost import ReferenceGhost
    from supermimic.robot import G1_JOINT_NAMES, TRACKED_BODIES
    from supermimic.rotations import mat_to_quat, quat_to_mat, so3_exp, yaw_mat

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
    try:
        scene = ViserMujocoScene(server, player.env.sim.mj_model, num_envs=1)
        scene.create_visualization_gui(camera_distance=3.5)
        data = mujoco.MjData(player.env.sim.mj_model)
        ghost = ReferenceGhost(server, player)
        player.step()
        sync_scene(player, scene, data)
        np.testing.assert_allclose(data.qpos, player.env.sim.data.qpos[0].cpu().numpy())
        assert np.isfinite(data.xpos).all()
        env = player.env
        player.seek(5)
        frontier = env.sim.data.qpos.clone()
        player.seek(1)
        snapshot = player.current_frame
        sync_scene(player, scene, data, snapshot)
        ghost.update(player, scene, data, align=False, frame=snapshot)
        np.testing.assert_array_equal(data.qpos, snapshot.qpos)
        np.testing.assert_allclose(
            ghost.data.qpos[ghost.root_qadr[:3]], env.motion.root_p[snapshot.reference_frame].cpu(), atol=1e-6,
        )
        np.testing.assert_allclose(
            ghost.data.qpos[ghost.joint_qadr], env.motion.q[snapshot.reference_frame].cpu(), atol=1e-6,
        )
        np.testing.assert_allclose(ghost.data.qpos[ghost.hand_qadr], snapshot.hand_q, atol=1e-6)
        torch.testing.assert_close(env.sim.data.qpos.clone(), frontier)
        player.seek(player.cached_steps)
        sync_scene(player, scene, data)
        # Introduce translation and yaw drift to exercise both ghost modes.
        p, R = env.state()[:2]
        p[:, :2] += torch.tensor([[1.0, -0.5]], device=env.device)
        R = so3_exp(torch.tensor([[0.0, 0.0, 0.7]], device=env.device)) @ R
        pose = torch.cat((p, mat_to_quat(R)), -1)
        env.robot.write_root_link_pose_to_sim(pose)
        env.refresh()
        sync_scene(player, scene, data)
        before = env.sim.data.qpos.clone()
        ghost.update(player, scene, data, align=False)
        assert ghost.meshes
        np.testing.assert_allclose(ghost.data.qpos[ghost.root_qadr[:3]], env.motion.root_p[env.f][0].cpu(), atol=1e-6)
        np.testing.assert_allclose(ghost.data.qpos[ghost.joint_qadr], env.motion.q[env.f][0].cpu(), atol=1e-6)
        np.testing.assert_allclose(ghost.data.qpos[ghost.hand_qadr], env.motion.hand_q[env.f][0].cpu(), atol=1e-6)
        tracked = [ghost.model.body(f"robot/{name}").id for name in TRACKED_BODIES]
        np.testing.assert_allclose(ghost.data.xpos[tracked], env.motion.body_p[env.f][0].cpu(), atol=1e-5)
        # Resolve joint values by name to catch SDK/MJCF ordering differences.
        for j, name in enumerate(G1_JOINT_NAMES):
            adr = ghost.model.joint(f"robot/{name}").qposadr[0]
            assert ghost.data.qpos[adr] == pytest.approx(float(env.motion.q[env.f, j]))
        ghost.update(player, scene, data, align=True, visible=False, opacity=0.45)
        np.testing.assert_allclose(ghost.data.qpos[ghost.root_qadr[:2]], p[0, :2].cpu(), atol=1e-6)
        ref_R = env.motion.root_R[env.f].cpu().double()
        policy_R = torch.tensor(data.xmat[ghost.root_body_id].reshape(1, 3, 3))
        expected_R = yaw_mat(policy_R @ ref_R.transpose(-1, -2)) @ ref_R
        actual_R = quat_to_mat(torch.tensor(ghost.data.qpos[ghost.root_qadr[3:]]))
        torch.testing.assert_close(actual_R, expected_R[0], atol=1e-6, rtol=1e-6)
        assert not ghost.frame.visible
        assert all(handle.opacity == 0.45 for handle in ghost.meshes.values())
        np.testing.assert_allclose(ghost.frame.position, -data.xpos[ghost.root_body_id])
        scene.camera_tracking_enabled = False
        ghost.update(player, scene, data)
        np.testing.assert_allclose(ghost.frame.position, np.zeros(3))
        torch.testing.assert_close(env.sim.data.qpos.clone(), before)
        connection = HTTPConnection("127.0.0.1", server.get_port(), timeout=5)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200
            assert b"<html" in response.read().lower()
        finally:
            connection.close()
    finally:
        server.stop()
