#!/usr/bin/env python3
"""Generate the phase-2 TRON2A training URDF from the frozen phase-1 source."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URDF_PATH = REPOSITORY_ROOT / "assets/resources/tron2a_10dof/urdf/robot_rl.urdf"
TRAINING_URDF_PATH = REPOSITORY_ROOT / "assets/resources/tron2a_10dof/urdf/tron2a_10dof_hold_arms.urdf"


def source_sha256() -> str:
    return hashlib.file_digest(SOURCE_URDF_PATH.open("rb"), "sha256").hexdigest()


def referenced_mesh_paths(urdf_path: Path) -> tuple[Path, ...]:
    root = ET.parse(urdf_path).getroot()
    return tuple(
        (urdf_path.parent / mesh.attrib["filename"]).resolve()
        for mesh in root.findall(".//mesh")
        if "filename" in mesh.attrib
    )


def generate_training_urdf() -> Path:
    root = ET.parse(SOURCE_URDF_PATH).getroot()
    root.attrib["name"] = "tron2a_10dof_hold_arms"
    root.insert(0, ET.Comment(f"Generated from robot_rl.urdf sha256={source_sha256()}"))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(TRAINING_URDF_PATH, encoding="utf-8", xml_declaration=True)
    return TRAINING_URDF_PATH


if __name__ == "__main__":
    print(generate_training_urdf())
