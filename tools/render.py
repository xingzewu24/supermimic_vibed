"""Render recorded evaluation trajectories (supermimic/evaluate.py --record) with MuJoCo (EGL).

The simulated robot is drawn solid; the reference motion is a translucent ghost.
Usage: MUJOCO_GL=egl python tools/render.py traj.npz out.mp4 --clips dance1_subject1 jumps1_subject1 --seconds 12
"""

import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont

from supermimic.robot import (
    ALL_HAND_JOINT_NAMES,
    G1_JOINT_NAMES,
    HAND_JOINT_NAMES,
    HAND_MIMIC_JOINT_NAMES,
    ROBOT_DIR,
    g1_spec,
    hand_joint_positions,
    mimic_relations,
    object_spec,
)


def two_robot_xml(object_path: str | None = None) -> str:
    """Attach both full assets through MjSpec so meshes and joint references stay valid."""
    spec = mujoco.MjSpec.from_string("""
    <mujoco>
    <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>
    <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85" width="512" height="512"/>
      <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
      <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.75 0.8 0.88" width="256" height="256"/>
    </asset>
    <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    </worldbody></mujoco>
    """)
    for prefix in ("robot/", "ref/"):
        robot = g1_spec()
        for mesh in robot.meshes:
            mesh.file = os.path.join(ROBOT_DIR, mesh.file)
        for geom in robot.geoms:
            geom.contype = geom.conaffinity = 0
            if geom.group != 1:
                geom.rgba[3] = 0.0  # collision proxies must not obscure either robot's visual meshes
            elif prefix == "ref/":
                geom.rgba = [0.35, 0.65, 1.0, 0.25]
        spec.attach(robot, prefix=prefix, frame=spec.worldbody.add_frame())
    if object_path is not None:
        for prefix in ("object/", "object_ref/"):
            obj = object_spec(object_path)
            for geom in obj.geoms:
                geom.contype = geom.conaffinity = 0
                if prefix == "object_ref/":
                    geom.rgba = [0.35, 0.65, 1.0, 0.25]
            spec.attach(obj, prefix=prefix, frame=spec.worldbody.add_frame())
    return spec.to_xml()


class TrajectoryPoseWriter:
    """Resolve every coordinate by joint name, including all recorded finger angles."""

    def __init__(self, model, trajectory):
        self.trajectory = trajectory
        self.root = {}
        self.body = {}
        self.hand = {}
        self.object = {}
        if "object_p" in trajectory:
            for prefix in ("object/", "object_ref/"):
                start = model.joint(prefix + "object_free_joint").qposadr[0]
                self.object[prefix] = np.arange(start, start + 7)
        for prefix in ("robot/", "ref/"):
            root_joint = model.joint(f"{prefix}floating_base_joint")
            self.root[prefix] = np.arange(root_joint.qposadr[0], root_joint.qposadr[0] + 7)
            self.body[prefix] = np.array([model.joint(prefix + n).qposadr[0] for n in G1_JOINT_NAMES])
            self.hand[prefix] = np.array([model.joint(prefix + n).qposadr[0] for n in ALL_HAND_JOINT_NAMES])
        pose = hand_joint_positions(trajectory.get("hand_target"))
        self.reference_hand = np.array([pose[n] for n in ALL_HAND_JOINT_NAMES])
        relations = mimic_relations()
        self.command_drivers = [HAND_JOINT_NAMES.index(relations[n][0]) for n in HAND_MIMIC_JOINT_NAMES]
        self.command_ratios = np.array([relations[n][1] for n in HAND_MIMIC_JOINT_NAMES])
        self.command_offsets = np.array([relations[n][2] for n in HAND_MIMIC_JOINT_NAMES])
        self.recorded_hand = None
        if "hand_q" in trajectory:
            names = list(trajectory["hand_joint_names"])
            self.recorded_hand = [names.index(n) for n in ALL_HAND_JOINT_NAMES]

    def write(self, data, t, clip, *, global_ghost=False):
        d = self.trajectory
        f = int(d["f"][t, clip])
        data.qpos[self.root["robot/"]] = np.concatenate((d["root_p"][t, clip], d["root_quat"][t, clip]))
        data.qpos[self.body["robot/"]] = d["q"][t, clip]
        data.qpos[self.hand["robot/"]] = (
            d["hand_q"][t, clip, self.recorded_hand] if self.recorded_hand is not None else self.reference_hand
        )
        p = d["ref_root_p"][f].copy()
        if not global_ghost:
            p[:2] = d["root_p"][t, clip, :2]
        data.qpos[self.root["ref/"]] = np.concatenate((p, _mat_to_quat(d["ref_root_R"][f])))
        data.qpos[self.body["ref/"]] = d["ref_q"][f]
        if "ref_hand_q" in d:
            data.qpos[self.hand["ref/"]] = d["ref_hand_q"][f]
        elif "hand_command" in d:  # Older recordings only; no live command source.
            command = d["hand_command"][t, clip]
            data.qpos[self.hand["ref/"]] = np.concatenate((
                command, command[self.command_drivers] * self.command_ratios + self.command_offsets,
            ))
        else:
            data.qpos[self.hand["ref/"]] = self.reference_hand
        if self.object:
            data.qpos[self.object["object/"]] = np.concatenate((d["object_p"][t, clip], d["object_quat"][t, clip]))
            data.qpos[self.object["object_ref/"]] = np.concatenate((d["ref_object_p"][f], _mat_to_quat(d["ref_object_R"][f])))
        mujoco.mj_kinematics(data.model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--global_ghost", action="store_true", help="draw the reference at its global xy instead of anchored to the robot")
    a = ap.parse_args()

    d = dict(np.load(a.traj))
    names = list(d["names"])
    model = mujoco.MjModel.from_xml_string(two_robot_xml(str(d["object_path"].item()) if "object_path" in d else None))
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    poses = TrajectoryPoseWriter(model, d)
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.6, 135.0, -12.0
    font = ImageFont.load_default(size=22)
    clips = a.clips or names
    ff = subprocess.Popen(
        [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    stride = round(50 / a.fps)
    for clip in clips:
        c = names.index(clip)
        T = d["root_p"].shape[0]
        alive = d["alive"][:, c]
        t0 = int(a.start * 50)
        t1 = min(T, t0 + int(a.seconds * 50))
        fell_at = None
        for t in range(t0, t1, stride):
            if not alive[t] and fell_at is None:
                fell_at = t
            tt = t if alive[t] else (fell_at if fell_at is not None else t)
            f = int(d["f"][tt, c])
            poses.write(data, tt, c, global_ghost=a.global_ghost or "object_p" in d)
            cam.lookat[:] = [*d["root_p"][tt, c, :2], 0.75]
            if "object_p" in d:
                root, ref_root = d["root_p"][tt, c], d["ref_root_p"][f]
                height = np.array([0.0, 0.0, 0.8])
                points = np.array([root - height, root + height, ref_root - height, ref_root + height,
                                   d["object_p"][tt, c], d["ref_object_p"][f]])
                cam.lookat[:] = (points.min(0) + points.max(0)) / 2
                cam.distance = max(3.6, 2.7 * np.linalg.norm(points - cam.lookat, axis=-1).max() + 0.5)
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            label = f"{a.title}  {clip}  t={(f - int(d['clip_start'][c])) * 0.02:5.1f}s"
            if fell_at is not None:
                label += "  [terminated]"
            for xy, txt in (((16, 12), label), ((16, a.height - 38), "solid: simulated G1 policy (mjlab + MuJoCo-Warp)   blue ghost: reference")):
                bb = draw.textbbox(xy, txt, font=font)
                draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                draw.text(xy, txt, fill=(20, 20, 20), font=font)
            ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


def _mat_to_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


if __name__ == "__main__":
    main()
