"""A visual-only blue reference robot in the live Viser scene."""

from __future__ import annotations

import mujoco
import numpy as np
from mjviser.conversions import merge_geoms

from .robot import ALL_HAND_JOINT_NAMES, G1_JOINT_NAMES


class ReferenceGhost:
    """Use independent CPU FK and Viser meshes; never enter the physics model."""

    def __init__(self, server, player):
        env = player.env
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.root_qadr = env.robot.indexing.free_joint_q_adr.cpu().numpy()
        self.joint_qadr = np.array([
            self.model.joint(f"robot/{name}").qposadr[0] for name in G1_JOINT_NAMES
        ])
        self.hand_qadr = np.array([
            self.model.joint(f"robot/{name}").qposadr[0] for name in ALL_HAND_JOINT_NAMES
        ])
        self.root_body_id = self.model.body("robot/pelvis").id
        self.object_qadr = env.object.indexing.free_joint_q_adr.cpu().numpy()
        self.frame = server.scene.add_frame("/reference", show_axes=False)
        self.meshes = {}
        robot_geoms = env.robot.indexing.geom_ids.cpu().numpy()
        for body_id in np.unique(self.model.geom_bodyid[robot_geoms]):
            # Group 1 contains the URDF visual meshes; omit collision proxies.
            ids = [int(g) for g in robot_geoms
                   if self.model.geom_bodyid[g] == body_id and self.model.geom_group[g] == 1
                   and self.model.geom_rgba[g, 3] > 0]
            if not ids:
                continue
            mesh = merge_geoms(self.model, ids)
            self.meshes[int(body_id)] = server.scene.add_mesh_simple(
                f"/reference/body_{body_id}", mesh.vertices, mesh.faces,
                color=(80, 155, 255), opacity=0.3, cast_shadow=False, receive_shadow=False,
            )
        object_geoms = env.object.indexing.geom_ids.cpu().numpy()
        for body_id in np.unique(self.model.geom_bodyid[object_geoms]):
            ids = [int(g) for g in object_geoms if self.model.geom_bodyid[g] == body_id]
            mesh = merge_geoms(self.model, ids)
            self.meshes[int(body_id)] = server.scene.add_mesh_simple(
                f"/reference/object_{body_id}", mesh.vertices, mesh.faces,
                color=(80, 155, 255), opacity=0.3, cast_shadow=False, receive_shadow=False,
            )

    def update(self, player, scene, policy_data, *, visible=True, align=True, opacity=0.3, frame=None) -> None:
        env, motion = player.env, player.env.motion
        f = env.f if frame is None else [frame.reference_frame]
        p = (motion.root_p[f] + env.scene.env_origins)[0].cpu().numpy().astype(np.float64)
        R = motion.root_R[f][0].cpu().numpy().astype(np.float64)
        if align:
            p[:2] = policy_data.qpos[self.root_qadr[:2]]
            policy_R = policy_data.xmat[self.root_body_id].reshape(3, 3)
            relative_R = policy_R @ R.T
            yaw = np.arctan2(relative_R[1, 0], relative_R[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, R.ravel())
        self.data.qpos[:] = policy_data.qpos
        self.data.qpos[self.root_qadr] = np.concatenate((p, quat))
        self.data.qpos[self.joint_qadr] = motion.q[f].cpu().numpy()[0]
        self.data.qpos[self.hand_qadr] = motion.hand_q[f][0].cpu().numpy()
        object_R = motion.object_R[f][0].cpu().numpy().astype(np.float64)
        object_quat = np.empty(4)
        mujoco.mju_mat2Quat(object_quat, object_R.ravel())
        self.data.qpos[self.object_qadr] = np.concatenate((
            (motion.object_p[f] + env.scene.env_origins)[0].cpu().numpy(), object_quat,
        ))
        mujoco.mj_kinematics(self.model, self.data)
        # mjviser centers all geometry on the policy root while camera tracking is on.
        self.frame.position = -policy_data.xpos[self.root_body_id] if scene.camera_tracking_enabled else np.zeros(3)
        self.frame.visible = visible
        for body_id, handle in self.meshes.items():
            handle.position = self.data.xpos[body_id]
            handle.wxyz = self.data.xquat[body_id]
            handle.opacity = opacity
