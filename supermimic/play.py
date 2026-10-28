"""Run checkpoint policies live on the server and view them through Viser."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event
from typing import TYPE_CHECKING

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from supermimic.checkpoint import load_policy
from supermimic.config import (
    DEFAULT_MOTION_DIR,
    HandControlConfig,
    ObjectTrackingConfig,
    PlayConfig,
    register_configs,
    validate_play_config,
)

register_configs()

if TYPE_CHECKING:
    import numpy as np


@dataclass
class PlaybackFrame:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    hand_q: np.ndarray
    reference_frame: int
    return_sum: float
    finished: bool
    failed: bool


class PolicyPlayer:
    """Run a policy from the clip start and cache its trajectory for seeking."""

    def __init__(self, env, policy, checkpoint_args: dict, clip: str | None = None):
        if env.N != 1 or not env.eval_mode:
            raise ValueError("play requires one evaluation environment")
        self.env = env
        self.policy = policy
        self.goal_frames = checkpoint_args["goal_frames"]
        self.hist_len = checkpoint_args["hist"]
        self.reset(clip or env.motion.names[0])

    def reset(self, clip: str) -> None:
        import torch

        if clip not in self.env.motion.names:
            raise ValueError(f"Unknown clip {clip!r}; available clips: {', '.join(self.env.motion.names)}")
        index = self.env.motion.names.index(clip)
        self.clip = clip
        self.last_frame = int(self.env.motion.clip_end[index] - self.env.motion.clip_start[index])
        self.history = None
        self.return_sum = 0.0
        self.finished = False
        self.failed = False
        self.steps = 0
        self.env.done_eval.zero_()
        self.env.failed_eval.zero_()
        self.env.reset(
            torch.zeros(1, dtype=torch.long, device=self.env.device),
            self.env.motion.clip_start[index:index + 1],
        )
        self._frames = [self._capture_frame()]

    @property
    def current_frame(self) -> PlaybackFrame:
        return self._frames[self.steps]

    @property
    def cached_steps(self) -> int:
        return len(self._frames) - 1

    def _capture_frame(self) -> PlaybackFrame:
        env = self.env
        return PlaybackFrame(
            qpos=env.sim.data.qpos[0].detach().cpu().numpy().copy(),
            qvel=env.sim.data.qvel[0].detach().cpu().numpy().copy(),
            ctrl=env.sim.data.ctrl[0].detach().cpu().numpy().copy(),
            hand_q=env.motion.hand_q[env.f][0].detach().cpu().numpy().copy(),
            reference_frame=int(env.f[0]),
            return_sum=self.return_sum, finished=self.finished, failed=self.failed,
        )

    def _show_frame(self, frame: int) -> None:
        self.steps = frame
        snapshot = self.current_frame
        self.return_sum = snapshot.return_sum
        self.finished, self.failed = snapshot.finished, snapshot.failed

    def seek(self, frame: int, *, max_steps: int | None = None) -> bool:
        """Seek a clip-relative frame; compute unseen frames without real-time waits.

        Physics and policy history stay at the cache frontier. Replaying cached
        frames only changes the display, so later inference follows the same run.
        A budget lets the viewer keep responding while computing a long seek.
        Return True when the frame is reached, or the policy terminates earlier.
        """
        if not 0 <= frame <= self.last_frame:
            raise ValueError(f"Frame must be in [0, {self.last_frame}], got {frame}")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        if frame <= self.cached_steps:
            self._show_frame(frame)
            return True
        self._show_frame(self.cached_steps)
        remaining = frame - self.steps if max_steps is None else max_steps
        for _ in range(min(frame - self.steps, remaining)):
            if self.finished:
                break
            self.step()
        return self.steps == frame or self.finished

    def step(self) -> None:
        import torch

        if self.finished:
            return
        if self.steps < self.cached_steps:
            self._show_frame(self.steps + 1)
            return
        with torch.no_grad():
            env = self.env
            obs = env.mdp.obs(env.state(), env.a_prev, env.xi, env.f)
            if self.history is None:
                self.history = obs.unsqueeze(1).repeat(1, self.hist_len, 1)
            else:
                self.history = torch.cat((self.history[:, 1:], obs.unsqueeze(1)), 1)
            action = self.policy(self.history, env.motion.goal(env.f, self.goal_frames))
            out = env.step(action)
            self.return_sum += float(out["r"][0])
            self.steps += 1
            self.finished = bool(env.done_eval[0])
            self.failed = bool(env.failed_eval[0])
            self._frames.append(self._capture_frame())


def create_player(cfg: DictConfig) -> PolicyPlayer:
    import torch

    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.motion import MotionLib
    from supermimic.robot import default_joint_pos_list

    path = Path(cfg.ckpt) if cfg.ckpt else Path(cfg.run_dir) / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}; set ckpt=<checkpoint path>")
    torch.cuda.set_device(cfg.device)
    checkpoint = torch.load(path, map_location=cfg.device)
    motion_dir = cfg.motion_dir or checkpoint["args"].get("motion_dir", DEFAULT_MOTION_DIR)
    kin = G1Kinematics(cfg.device)
    clips = checkpoint["args"].get("motion_clips") if cfg.motion_dir is None else None
    motion = MotionLib.from_dir(motion_dir, kin, cfg.device, **({"clips": clips} if clips is not None else {}))
    if cfg.motion_dir is None and checkpoint["args"].get("motion_signature") != motion.signature:
        raise ValueError("Checkpoint motion/asset signature changed; set motion_dir explicitly to play new data")
    if cfg.clip is not None and cfg.clip not in motion.names:
        raise ValueError(f"Unknown clip {cfg.clip!r}; available clips: {', '.join(motion.names)}")
    policy = load_policy(checkpoint, cfg.device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=cfg.device),
              hand_cfg=HandControlConfig(**checkpoint["args"]["hand"]),
              object_cfg=ObjectTrackingConfig(**checkpoint["args"]["object"]))
    env = TrackingSim(make_sim(cfg.device, cfg), 1, motion, mdp, eval_mode=True)
    return PolicyPlayer(env, policy, checkpoint["args"], cfg.clip)


def sync_scene(player: PolicyPlayer, scene, data, frame: PlaybackFrame | None = None) -> None:
    """Copy integrated qpos, then run CPU FK so the display has no one-step lag."""
    import mujoco

    if frame is None:
        data.qpos[:] = player.env.sim.data.qpos[0].cpu().numpy()
        data.qvel[:] = player.env.sim.data.qvel[0].cpu().numpy()
        data.ctrl[:] = player.env.sim.data.ctrl[0].cpu().numpy()
    else:
        data.qpos[:], data.qvel[:], data.ctrl[:] = frame.qpos, frame.qvel, frame.ctrl
    mujoco.mj_forward(player.env.sim.mj_model, data)
    scene.update_from_mjdata(data)


def play_run(cfg: DictConfig) -> None:
    import mujoco
    import viser
    from mjviser import ViserMujocoScene

    from supermimic.ghost import ReferenceGhost
    from supermimic.motion import CONTROL_DT

    player = create_player(cfg)
    server = viser.ViserServer(host=cfg.host, port=cfg.port, label="SuperMimic")
    try:
        scene = ViserMujocoScene(server, player.env.sim.mj_model, num_envs=1)
        data = mujoco.MjData(player.env.sim.mj_model)
        ghost = ReferenceGhost(server, player)
        reset_requested, step_requested = Event(), Event()
        seek_requests = SimpleQueue()
        with server.gui.add_folder("Playback"):
            clips = server.gui.add_dropdown("Motion", options=player.env.motion.names, initial_value=player.clip)
            paused = server.gui.add_checkbox("Paused", initial_value=cfg.paused)
            timeline = server.gui.add_slider(
                "Frame", min=0, max=player.last_frame, step=1, initial_value=0,
                hint="Drag to seek. 50 frames = 1 second. Unseen frames are simulated as fast as possible.",
            )
            speed = server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=cfg.speed)
            loop = server.gui.add_checkbox("Loop", initial_value=cfg.loop)
            reset = server.gui.add_button("Restart motion")
            previous = server.gui.add_button("Previous frame")
            step = server.gui.add_button("Single step")
            back = server.gui.add_button("Back 5 s")
            forward = server.gui.add_button("Forward 5 s")
            status = server.gui.add_markdown("")
        with server.gui.add_folder("Reference motion"):
            show_reference = server.gui.add_checkbox("Show blue reference", initial_value=cfg.show_reference)
            align_reference = server.gui.add_checkbox("Align XY and yaw to policy", initial_value=cfg.align_reference)
            reference_opacity = server.gui.add_slider(
                "Opacity", min=0.0, max=1.0, step=0.05, initial_value=cfg.reference_opacity,
            )
            server.gui.add_markdown("Solid: policy · Blue ghost: reference (same frame).")

        @reset.on_click
        def _(_event):
            reset_requested.set()

        @step.on_click
        def _(_event):
            paused.value = True
            step_requested.set()

        @timeline.on_update
        def _(event):
            # Server-side progress updates also fire callbacks.
            if event.client_id is not None:
                paused.value = True
                seek_requests.put((clips.value, int(event.target.value)))

        def request_offset(offset):
            paused.value = True
            seek_requests.put((clips.value, int(timeline.value) + offset))

        @previous.on_click
        def _(_event):
            request_offset(-1)

        @back.on_click
        def _(_event):
            request_offset(-round(5 / CONTROL_DT))

        @forward.on_click
        def _(_event):
            request_offset(round(5 / CONTROL_DT))

        separation = (player.env.motion.object_p - player.env.motion.root_p).norm(dim=-1).max().item()
        scene.create_visualization_gui(camera_distance=max(3.5, 2.8 * separation + 1.0), camera_elevation=15.0)
        port = server.get_port()
        print(f"[play] Viser ready on {cfg.host}:{port}. Physics: 50 Hz; render: {cfg.fps} FPS.", flush=True)
        print(f"[play] On your computer: ssh -N -L {port}:127.0.0.1:{port} USER@SERVER", flush=True)
        print(f"[play] Then open http://localhost:{port}. Waiting for a browser connection.", flush=True)
        next_step = next_render = 0.0
        finished_at = None
        dirty = True
        previous_controls = None
        seek_target = None
        while True:
            now = time.monotonic()
            controls = (
                paused.value, speed.value, loop.value,
                show_reference.value, align_reference.value, reference_opacity.value,
            )
            if controls != previous_controls:
                dirty = True
                previous_controls = controls
            if reset_requested.is_set() or clips.value != player.clip:
                reset_requested.clear()
                player.reset(clips.value)
                seek_target = None
                timeline.max = player.last_frame
                timeline.value = 0
                step_requested.clear()
                while not seek_requests.empty():
                    seek_requests.get_nowait()
                finished_at = None
                dirty = True
                next_step = now
            try:
                while True:
                    clip, frame = seek_requests.get_nowait()
                    if clip == player.clip:
                        seek_target = max(0, min(frame, player.last_frame))
            except Empty:
                pass
            single_step = step_requested.is_set()
            step_requested.clear()
            connected = bool(server.get_clients())
            if single_step:
                seek_target = None
            if seek_target is not None and connected:
                reached = player.seek(seek_target, max_steps=8)
                dirty = True
                next_step = time.monotonic()
                if reached:
                    seek_target = None
                    finished_at = time.monotonic() if player.finished else None
            if not connected or paused.value:
                next_step = now
            if single_step or (connected and not paused.value and seek_target is None and now >= next_step):
                if player.finished:
                    if single_step or (loop.value and finished_at is not None and now - finished_at >= 1.0):
                        player.seek(0)
                        finished_at = None
                        dirty = True
                    elif not loop.value:
                        paused.value = True
                if not player.finished:
                    player.step()
                    dirty = True
                    if player.finished:
                        finished_at = time.monotonic()
                # Fall behind gracefully if inference/physics cannot keep up with the requested speed.
                next_step = max(next_step + CONTROL_DT / speed.value, time.monotonic())
            if now >= next_render and (dirty or scene.needs_update):
                with server.atomic():
                    sync_scene(player, scene, data, player.current_frame)
                    ghost.update(
                        player, scene, data, visible=show_reference.value,
                        align=align_reference.value, opacity=reference_opacity.value,
                        frame=player.current_frame,
                    )
                    state = (
                        f"Seeking frame {seek_target}" if seek_target is not None else
                        "Failed" if player.failed else "Completed" if player.finished
                        else "Paused" if paused.value or not connected else "Playing"
                    )
                    status.content = (
                        f"**{player.clip}** · {state}  \n"
                        f"Frame: {player.steps}/{player.last_frame} · "
                        f"Time: {player.steps * CONTROL_DT:.2f} s · Return: {player.return_sum:.3f}"
                    )
                    timeline.value = player.steps
                dirty = False
                next_render = max(next_render + 1.0 / cfg.fps, time.monotonic())
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


@hydra.main(version_base="1.3", config_path="conf", config_name="play")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(PlayConfig), cfg)
    OmegaConf.resolve(cfg)
    for key in ("run_dir", "ckpt", "motion_dir"):
        if cfg[key] is not None:
            cfg[key] = to_absolute_path(cfg[key])
    validate_play_config(cfg)
    play_run(cfg)


if __name__ == "__main__":
    main()
