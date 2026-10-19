#!/usr/bin/env python3
"""Validate the phase-1 TRON2A asset, joint, and motion contracts offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "assets/resources/tron2a_10dof/MANIFEST.json"
CONTRACT_PATH = REPOSITORY_ROOT / "tasks/tron2a_10dof_parkour/data/joint_contract.json"
LOCKED_POSE_PATH = REPOSITORY_ROOT / "tasks/tron2a_10dof_parkour/data/locked_joint_pose.yaml"


class ContractError(ValueError):
    """An immutable phase-1 input no longer matches its declared contract."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = {artifact["id"]: artifact for artifact in manifest["artifacts"]}
    if len(artifacts) != len(manifest["artifacts"]):
        raise ContractError("MANIFEST.json contains duplicate artifact ids")
    return artifacts


def parse_movable_joints(urdf_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        if joint.attrib["type"] == "fixed":
            continue
        name = joint.attrib["name"]
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        if parent is None or child is None or axis is None or limit is None:
            raise ContractError(f"{name}: movable URDF joint misses parent, child, axis, or limit")
        child_name = child.attrib["link"]
        inertial = links[child_name].find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        inertia = inertial.find("inertia") if inertial is not None else None
        if mass is None or inertia is None:
            raise ContractError(f"{name}: child link {child_name} misses inertial data")
        joints.append(
            {
                "name": name,
                "type": joint.attrib["type"],
                "parent_link": parent.attrib["link"],
                "child_link": child_name,
                "axis_xyz": [float(value) for value in axis.attrib["xyz"].split()],
                "limit": {key: float(limit.attrib[key]) for key in ("lower", "upper", "effort", "velocity")},
                "child_inertial": {
                    "mass": float(mass.attrib["value"]),
                    "inertia": {key: float(value) for key, value in inertia.attrib.items()},
                },
            }
        )
    return joints


def validate_contract() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    contract = load_json(CONTRACT_PATH)
    # JSON is a YAML 1.2 subset, allowing this safety-critical config to remain dependency-free.
    locked_pose = load_json(LOCKED_POSE_PATH)
    artifacts = artifact_map(manifest)
    errors: list[str] = []

    resolved_artifacts: dict[str, str] = {}
    for artifact in manifest["artifacts"]:
        source = Path(artifact["source_path"])
        workspace = (MANIFEST_PATH.parent / artifact["workspace_path"]).resolve()
        if not source.exists():
            errors.append(f"{artifact['id']}: source does not exist: {source}")
            continue
        if not workspace.exists():
            errors.append(f"{artifact['id']}: workspace link does not resolve: {workspace}")
        elif workspace != source.resolve():
            errors.append(f"{artifact['id']}: workspace link resolves to {workspace}, expected {source.resolve()}")
        if "sha256" in artifact and source.is_file():
            actual_hash = sha256(source)
            if actual_hash != artifact["sha256"]:
                errors.append(f"{artifact['id']}: SHA256 mismatch: {actual_hash}")
        resolved_artifacts[artifact["id"]] = str(source.resolve())

    urdf = Path(artifacts[contract["source_urdf_artifact"]]["source_path"])
    joints = parse_movable_joints(urdf)
    urdf_names = [joint["name"] for joint in joints]
    all_joints = contract["all_joints_18"]
    policy_joints = contract["policy_joints_10"]
    locked_joints = contract["locked_joints_8"]

    if urdf_names != all_joints:
        errors.append("URDF movable-joint order differs from all_joints_18")
    if len(urdf_names) != 18:
        errors.append(f"URDF contains {len(urdf_names)} movable joints, expected 18")
    if len(policy_joints) != 10:
        errors.append(f"policy_joints_10 contains {len(policy_joints)} joints, expected 10")
    if len(locked_joints) != 8:
        errors.append(f"locked_joints_8 contains {len(locked_joints)} joints, expected 8")
    if len(set(all_joints)) != len(all_joints):
        errors.append("all_joints_18 contains duplicates")
    if set(policy_joints) & set(locked_joints):
        errors.append("policy and locked joint sets overlap")
    if set(policy_joints) | set(locked_joints) != set(all_joints):
        errors.append("policy and locked joint sets do not cover all_joints_18")

    locked_entries = locked_pose["joints"]
    if set(locked_entries) != set(locked_joints):
        errors.append("locked_joint_pose.yaml does not define exactly locked_joints_8")
    joint_by_name = {joint["name"]: joint for joint in joints}
    for name in locked_joints:
        entry = locked_entries.get(name)
        if entry is None or name not in joint_by_name:
            continue
        target = entry["target_position"]
        limit = joint_by_name[name]["limit"]
        if not limit["lower"] <= target <= limit["upper"]:
            errors.append(f"{name}: lock target {target} is outside [{limit['lower']}, {limit['upper']}]")
        if entry["position_tolerance"] <= 0 or entry["stiffness"] <= 0 or entry["damping"] <= 0:
            errors.append(f"{name}: tolerance, stiffness, and damping must be positive")

    motion_spec = contract["npz_contract"]
    motion = Path(artifacts[motion_spec["artifact"]]["source_path"])
    with np.load(motion, allow_pickle=False) as data:
        required = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
        missing = required - set(data.files)
        if missing:
            errors.append(f"motion NPZ misses keys: {sorted(missing)}")
        else:
            motion_joint_names = data["joint_names"].tolist()
            joint_pos = data["joint_pos"]
            if motion_joint_names != all_joints:
                errors.append("motion joint_names differs from all_joints_18")
            if joint_pos.shape != (motion_spec["frame_count"], motion_spec["joint_count"]):
                errors.append(f"joint_pos shape is {joint_pos.shape}, expected {(motion_spec['frame_count'], motion_spec['joint_count'])}")
            if float(data["framerate"]) != motion_spec["framerate_hz"]:
                errors.append(f"motion framerate is {float(data['framerate'])}, expected {motion_spec['framerate_hz']}")
            if data["base_pos_w"].shape != (motion_spec["frame_count"], 3):
                errors.append("base_pos_w shape is invalid")
            if data["base_quat_w"].shape != (motion_spec["frame_count"], 4):
                errors.append("base_quat_w shape is invalid")

    report = {
        "status": "passed" if not errors else "failed",
        "robot": manifest["robot"],
        "movable_joint_count": len(urdf_names),
        "policy_joint_count": len(policy_joints),
        "locked_joint_count": len(locked_joints),
        "all_joint_order": urdf_names,
        "policy_joint_order": policy_joints,
        "locked_joint_order": locked_joints,
        "locked_joint_targets": {
            name: locked_entries[name]["target_position"] for name in locked_joints if name in locked_entries
        },
        "locked_pose_approval_status": locked_pose["approval_status"],
        "resolved_artifacts": resolved_artifacts,
        "parsed_urdf_joints": joints,
        "errors": errors,
    }
    if errors:
        raise ContractError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write the successful JSON report to this path.")
    args = parser.parse_args()
    try:
        report = validate_contract()
    except ContractError as error:
        print(error, file=sys.stderr)
        return 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
