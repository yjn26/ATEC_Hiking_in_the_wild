"""TRON2A 10-DoF training asset definition for phase 2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import xml.etree.ElementTree as ET

from assets.official_policy import policy_pd_gains


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "tasks/tron2a_10dof_parkour/data"
TRAINING_URDF_PATH = REPOSITORY_ROOT / "assets/resources/tron2a_10dof/urdf/tron2a_10dof_hold_arms.urdf"
MJCF_PATH = REPOSITORY_ROOT / "assets/resources/tron2a_10dof/mjcf/robot.xml"

_INITIAL_JOINT_POSITIONS = {
    "proximal_pitch_L_Joint": 0.0,
    "proximal_roll_L_Joint": 0.0,
    "proximal_yaw_L_Joint": -3.14159,
    "knee_L_Joint": 0.0,
    "ankle_pitch_L_Joint": 0.0,
    "proximal_pitch_R_Joint": 0.0,
    "proximal_roll_R_Joint": 0.0,
    "proximal_yaw_R_Joint": 3.14159,
    "knee_R_Joint": 0.0,
    "ankle_pitch_R_Joint": 0.0,
}


@dataclass(frozen=True)
class ActuatorSpec:
    effort_limit: float
    velocity_limit: float
    armature: float
    stiffness: float
    damping: float
    target_position: float | None = None


@dataclass(frozen=True)
class TrainingAssetSpec:
    urdf_path: Path
    root_link: str
    initial_root_position_m: float
    initial_joint_positions: Mapping[str, float]
    policy_actuators: Mapping[str, ActuatorSpec]
    locked_actuators: Mapping[str, ActuatorSpec]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _urdf_limits() -> dict[str, tuple[float, float]]:
    root = ET.parse(TRAINING_URDF_PATH).getroot()
    return {
        joint.attrib["name"]: (float(joint.find("limit").attrib["effort"]), float(joint.find("limit").attrib["velocity"]))
        for joint in root.findall("joint")
        if joint.attrib["type"] != "fixed"
    }


def _mjcf_armatures() -> dict[str, float]:
    root = ET.parse(MJCF_PATH).getroot()
    return {
        joint.attrib["name"]: float(joint.attrib["armature"])
        for joint in root.findall(".//joint")
        if "name" in joint.attrib and "armature" in joint.attrib
    }


def _leg_gains(name: str) -> tuple[float, float]:
    return policy_pd_gains(name)


def load_training_asset_spec() -> TrainingAssetSpec:
    contract = _load_json(DATA_ROOT / "joint_contract.json")
    locked_pose = _load_json(DATA_ROOT / "locked_joint_pose.yaml")
    limits = _urdf_limits()
    armatures = _mjcf_armatures()
    policy = {}
    for name in contract["policy_joints_10"]:
        stiffness, damping = _leg_gains(name)
        effort, velocity = limits[name]
        policy[name] = ActuatorSpec(effort, velocity, armatures[name], stiffness, damping)
    locked = {}
    for name in contract["locked_joints_8"]:
        effort, velocity = limits[name]
        pose = locked_pose["joints"][name]
        locked[name] = ActuatorSpec(
            effort, velocity, armatures[name], pose["stiffness"], pose["damping"], pose["target_position"]
        )
    return TrainingAssetSpec(
        urdf_path=TRAINING_URDF_PATH,
        root_link="base_Link",
        initial_root_position_m=0.85,
        initial_joint_positions=MappingProxyType(_INITIAL_JOINT_POSITIONS | {
            name: entry.target_position for name, entry in locked.items() if entry.target_position is not None
        }),
        policy_actuators=MappingProxyType(policy),
        locked_actuators=MappingProxyType(locked),
    )


def make_isaac_articulation_cfg():
    """Create the Isaac Lab articulation config only in an Isaac Lab runtime."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import DelayedPDActuatorCfg
    from isaaclab.assets import ArticulationCfg

    spec = load_training_asset_spec()
    all_actuators = spec.policy_actuators | spec.locked_actuators
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(spec.urdf_path),
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=True,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spec.initial_root_position_m),
            joint_pos=dict(spec.initial_joint_positions),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "legs": DelayedPDActuatorCfg(
                joint_names_expr=list(spec.policy_actuators),
                effort_limit={name: actuator.effort_limit for name, actuator in spec.policy_actuators.items()},
                velocity_limit={name: actuator.velocity_limit for name, actuator in spec.policy_actuators.items()},
                stiffness={name: actuator.stiffness for name, actuator in spec.policy_actuators.items()},
                damping={name: actuator.damping for name, actuator in spec.policy_actuators.items()},
                armature={name: actuator.armature for name, actuator in spec.policy_actuators.items()},
                min_delay=0,
                max_delay=0,
            ),
            "locked": DelayedPDActuatorCfg(
                joint_names_expr=list(spec.locked_actuators),
                effort_limit={name: actuator.effort_limit for name, actuator in spec.locked_actuators.items()},
                velocity_limit={name: actuator.velocity_limit for name, actuator in spec.locked_actuators.items()},
                stiffness={name: actuator.stiffness for name, actuator in spec.locked_actuators.items()},
                damping={name: actuator.damping for name, actuator in spec.locked_actuators.items()},
                armature={name: actuator.armature for name, actuator in spec.locked_actuators.items()},
                min_delay=0,
                max_delay=0,
            ),
        },
    )
