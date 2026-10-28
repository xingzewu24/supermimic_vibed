"""Full G1 Inspire articulation with 41 actions and 53 position actuators."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROBOT_DIR = os.path.join(DATA_DIR, "tracking", "robot", "g1_inspire")
MJCF_PATH = os.path.join(ROBOT_DIR, "g1_inspire.xml")
URDF_PATH = os.path.join(ROBOT_DIR, "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")
MOTION_DIR = os.path.join(DATA_DIR, "hoi", "motions", "test")
LAFAN_MOTION_DIR = os.path.join(DATA_DIR, "tracking", "motions", "lafan_g1_inspire")
RAW_MOTION_DIR = os.path.join(DATA_DIR, "tracking", "motions", "lafan1_raw")

HAND_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("L", "R")
    for joint in ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal",
                  "middle_proximal", "ring_proximal", "pinky_proximal")
]
HAND_MIMIC_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("L", "R")
    for joint in ("thumb_intermediate", "thumb_distal", "index_intermediate",
                  "middle_intermediate", "ring_intermediate", "pinky_intermediate")
]
ALL_HAND_JOINT_NAMES = HAND_JOINT_NAMES + HAND_MIMIC_JOINT_NAMES
# Aligned with g1_hoi_learning/robots/g1_inspire.py, including the followers.
# These are simulation parameters, not identified hardware parameters.
HAND_STIFFNESS = 5.0
HAND_DAMPING = 0.5
HAND_ARMATURE = 0.01
HAND_EFFORT_LIMIT = 1.0


@lru_cache
def urdf_joints() -> dict:
    return {j.attrib["name"]: j for j in ET.parse(URDF_PATH).getroot().findall("joint")}


def mimic_relations() -> dict[str, tuple[str, float, float]]:
    """Follower -> (driver, multiplier, offset), in actual joint coordinates."""
    return {
        name: (m.attrib["joint"], float(m.get("multiplier", 1)), float(m.get("offset", 0)))
        for name, joint in urdf_joints().items() if (m := joint.find("mimic")) is not None
    }


@lru_cache
def hand_joint_limits() -> tuple[list[float], list[float]]:
    """Driver ranges intersected with both assets and every follower's range."""
    xml = {j.attrib["name"]: j for j in ET.parse(MJCF_PATH).iter("joint") if "name" in j.attrib}
    ranges = {}
    for name in ALL_HAND_JOINT_NAMES:
        limit = urdf_joints()[name].find("limit")
        lo, hi = map(float, xml[name].attrib["range"].split())
        ranges[name] = (max(lo, float(limit.attrib["lower"])), min(hi, float(limit.attrib["upper"])))
    for follower, (driver, multiplier, offset) in mimic_relations().items():
        lo, hi = ranges[follower]
        if multiplier == 0:
            if not lo <= offset <= hi:
                raise ValueError(f"Infeasible mimic offset for {follower}")
            continue
        a, b = sorted(((lo - offset) / multiplier, (hi - offset) / multiplier))
        dlo, dhi = ranges[driver]
        ranges[driver] = (max(dlo, a), min(dhi, b))
    if any(ranges[n][0] >= ranges[n][1] for n in HAND_JOINT_NAMES):
        raise ValueError("Empty Inspire hand control range")
    return [ranges[n][0] for n in HAND_JOINT_NAMES], [ranges[n][1] for n in HAND_JOINT_NAMES]


def hand_joint_positions(target=None) -> dict[str, float]:
    """Validate all 24 hand angles against both assets, expanding the 12 targets."""
    target = [0.0] * 12 if target is None else list(target)
    if len(target) != 12 or any(not math.isfinite(v) for v in target):
        raise ValueError("hand_target must contain 12 finite angles in radians")
    positions = dict(zip(HAND_JOINT_NAMES, map(float, target)))
    relations = mimic_relations()
    if set(relations) != set(HAND_MIMIC_JOINT_NAMES):
        raise ValueError("Expected the 12 Inspire hand mimic relations")
    for follower, (driver, multiplier, offset) in relations.items():
        positions[follower] = multiplier * positions[driver] + offset
    xml_joints = {j.attrib["name"]: j for j in ET.parse(MJCF_PATH).iter("joint") if "name" in j.attrib}
    for name, value in positions.items():
        limit = urdf_joints()[name].find("limit")
        ranges = [(float(limit.attrib["lower"]), float(limit.attrib["upper"])),
                  tuple(map(float, xml_joints[name].attrib["range"].split()))]
        for lo, hi in ranges:
            if not lo <= value <= hi:
                raise ValueError(f"hand_target puts {name}={value:g} outside [{lo:g}, {hi:g}]")
    return positions

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _kp(arm):
    return arm * NATURAL_FREQ**2


def _kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


# Joint order of the LAFAN1 retarget CSVs (Unitree SDK order).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

# Per-joint (effort limit, velocity limit, armature) by motor type.
_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (35.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (35.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (35.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (35.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}


def motor_of(joint_name: str) -> tuple[float, float, float]:
    key = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
    return JOINT_MOTORS[key]


def action_scale(joint_name: str) -> float:
    effort, _, arm = motor_of(joint_name)
    return 0.25 * effort / _kp(arm)


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def default_joint_pos_list() -> list[float]:
    import re

    out = []
    for name in G1_JOINT_NAMES:
        val = 0.0
        for pat, v in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pat, name):
                val = v
        out.append(val)
    return out


def g1_spec() -> mujoco.MjSpec:
    """Load the full asset and supply URDF mimic constraints omitted by its XML."""
    for path in (MJCF_PATH, URDF_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}. Supply the full G1 Inspire XML, URDF and meshes in {ROBOT_DIR}.")
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    for follower, (driver, multiplier, offset) in mimic_relations().items():
        spec.add_equality(
            name=f"mimic_{follower}", type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=follower, name2=driver,
            data=[offset, multiplier, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            solref=(0.01, 1), solimp=(0.9999, 0.9999, 0.001, 0.5, 2),
        )
    for name in G1_JOINT_NAMES:
        effort = motor_of(name)[0]
        asset_effort = float(urdf_joints()[name].find("limit").attrib["effort"])
        if effort != asset_effort:
            raise ValueError(f"Body effort limit differs from URDF for {name}")
        spec.joint(name).actfrcrange = [-effort, effort]
    for geom in spec.geoms:
        # URDF visuals compile as group 1 with both collision masks zero.
        if geom.contype or geom.conaffinity:
            geom.contype = 0
            geom.conaffinity = 1 | 4  # ground and object; no robot self collisions
            geom.condim = 3
            geom.friction[0] = 1.0
    return spec


def get_g1_cfg() -> EntityCfg:
    hand_positions = hand_joint_positions()
    actuators = []
    for name in G1_JOINT_NAMES:
        effort, _, arm = motor_of(name)
        actuators.append(BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            effort_limit=effort,
            stiffness=_kp(arm),
            damping=_kd(arm),
            armature=arm,
        ))
    for name in ALL_HAND_JOINT_NAMES:
        if float(urdf_joints()[name].find("limit").attrib["effort"]) != HAND_EFFORT_LIMIT:
            raise ValueError(f"Hand effort limit differs from URDF for {name}")
        actuators.append(BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            effort_limit=HAND_EFFORT_LIMIT,
            stiffness=HAND_STIFFNESS, damping=HAND_DAMPING, armature=HAND_ARMATURE,
        ))
    return EntityCfg(
        spec_fn=g1_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76), joint_pos={**DEFAULT_JOINT_POS, **hand_positions}, joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9,
        ),
    )


def object_spec(path: str) -> mujoco.MjSpec:
    """Keep the supplied mesh, density, inertia and reference origin unchanged."""
    spec = mujoco.MjSpec.from_file(path)
    for mesh in spec.meshes:
        mesh.file = os.path.join(os.path.dirname(path), mesh.file)
    return spec


def get_object_cfg(path: str) -> EntityCfg:
    return EntityCfg(spec_fn=lambda: object_spec(path), init_state=EntityCfg.InitialStateCfg(pos=(0, 0, 1)))
