#!/usr/bin/env python3
"""Regenerate the MuJoCo cobblestone bed used by the terrain-selector scene.

The bed that ships in ``assets/scene/generated/`` is produced by the **same** generator
that builds the Isaac Sim trough (``pebble_scene/gen_pebbles.py``): one shape pool of 32
noise-deformed convex cobbles, the same stacking rules, the same density/friction values.
Only the budget differs — MuJoCo solves contacts on the CPU, so the basin gets ~1.2 k
large cobbles instead of ~18 k small ones (see the PR description for measurements).

This script is a thin wrapper: it finds the scene generator in the repo, runs the MuJoCo
exporter, and drops the result next to the scene. Run it from anywhere:

    python Hiking_deploy/sim2sim/generate_pebble_terrain.py

It writes:

    assets/scene/generated/pebble_field_v3.xml          bodies (include inside <worldbody>)
    assets/scene/generated/pebble_field_v3_assets.xml   mesh assets (include at <mujoco>)
    assets/scene/generated/meshes/pebble_s??.obj        the 32 unit cobble shapes

MJCF gotcha that used to bite here: ``<include>`` inserts the file's children into the
**parent** element, so mesh assets and bodies can not live in one include unless that
include sits at ``<mujoco>`` level — hence the two files.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "assets", "scene", "generated")

CANDIDATES = [
    os.path.join(REPO, "assets", "resources", "pebble_scene"),      # where the scene package lives
    os.path.join(REPO, "pebble_scene"),
    os.environ.get("PEBBLE_SCENE_DIR", ""),
]


def find_scene_dir():
    for c in CANDIDATES:
        if c and os.path.isfile(os.path.join(c, "export_mujoco.py")):
            return c
    sys.exit(
        "pebble_scene/ not found -- it ships with the scene package (PR 'pebble scene handoff').\n"
        "Searched:\n  " + "\n  ".join(c for c in CANDIDATES if c) + "\n"
        "Set PEBBLE_SCENE_DIR=/path/to/pebble_scene and retry."
    )


def main():
    scene = find_scene_dir()
    os.makedirs(OUT, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(scene, "export_mujoco.py"),
        "--config", os.path.join(scene, "config_mujoco_basin.json"),
        "--out", OUT,
        "--name", "pebble_field_v3",
        # the basin of scene_29dof_terrain_selector.xml: centre (-5.5, 5.5), floor top z = 0,
        # inner half-width 1.42 m, parapet 0.30 m
        "--footprint", "2.84", "2.84",
        "--offset", "-5.5", "5.5",
        "--collision", "ellipsoid",     # analytic contact; real-shaped mesh stays visual-only
        # asset paths resolve from the MAIN model (assets/scene/scene_*.xml), not from this include
        "--mesh-prefix", "generated/meshes",
        "--joints", "free",
    ]
    print("[pebble] " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=scene)
    print("[pebble] done -> " + OUT)


if __name__ == "__main__":
    main()
