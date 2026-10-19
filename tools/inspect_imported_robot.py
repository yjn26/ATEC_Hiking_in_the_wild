#!/usr/bin/env python3
"""Inspect the phase-2 training asset before or inside an Isaac Lab runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from assets.official_policy import (
    DEFAULT_POLICY_ROOT,
    OfficialOnnxPolicy,
    load_official_policy_spec,
    make_joint_position_targets,
    runtime_joint_ids,
)
from assets.tron2a_10dof import load_training_asset_spec, make_isaac_articulation_cfg
from tools.generate_training_urdf import referenced_mesh_paths


RUNTIME_ARTIFACT_PATH = REPOSITORY_ROOT / "artifacts/validation/tron2a-phase2-runtime.json"


def runtime_validation_contract() -> dict[str, float]:
    return {
        "duration_s": 10.0,
        "physics_dt_s": 0.005,
        "maximum_locked_joint_error": 0.2,
    }


def inspect_offline_training_asset() -> dict[str, object]:
    spec = load_training_asset_spec()
    root = ET.parse(spec.urdf_path).getroot()
    movable = [joint for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]
    missing_meshes = [str(path) for path in referenced_mesh_paths(spec.urdf_path) if not path.exists()]
    report = {
        "status": "passed" if not missing_meshes else "failed",
        "root_link": spec.root_link,
        "movable_joint_count": len(movable),
        "policy_joint_count": len(spec.policy_actuators),
        "locked_joint_count": len(spec.locked_actuators),
        "missing_meshes": missing_meshes,
    }
    if report["status"] != "passed":
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    return report


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_isaac_runtime_validation(
    headless: bool,
    artifact_path: Path,
    use_official_policy: bool = False,
    policy_root: Path = DEFAULT_POLICY_ROOT,
    duration_s: float | None = None,
    cam_track: bool = False,
    record_mp4: Path | None = None,
    capture_size: str = "1920x1080",
) -> dict[str, Any]:
    """Import the articulation and prove its 10-second static-hold contract in Isaac Sim."""
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as error:
        raise RuntimeError("Isaac Lab is required for GUI/headless import and 10-second hold validation") from error

    contract = runtime_validation_contract()
    if duration_s is not None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        contract["duration_s"] = duration_s
    if (cam_track or record_mp4 is not None) and headless:
        raise ValueError("cam_track and recording require a GUI session")
    frame_directory = None
    ffmpeg_binary = None
    if record_mp4 is not None:
        ffmpeg_binary = Path(os.environ.get("FFMPEG_BINARY", "/home/pc/miniconda3/bin/ffmpeg"))
        if not ffmpeg_binary.is_file():
            raise RuntimeError(f"ffmpeg binary is unavailable: {ffmpeg_binary}")
        record_mp4.parent.mkdir(parents=True, exist_ok=True)
        frame_directory = record_mp4.with_suffix("").with_name(f"{record_mp4.stem}_frames")
        frame_directory.mkdir(parents=True, exist_ok=True)
    app = AppLauncher({"headless": headless}).app
    simulation_context = None
    phase = "app_launched"
    diagnostic: dict[str, Any] | None = None
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import quat_apply_inverse
        if frame_directory is not None:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("Isaac Sim GUI viewport is unavailable for recording")

        phase = "runtime_imports_ready"
        spec = load_training_asset_spec()
        policy_spec = load_official_policy_spec()
        policy_runner = OfficialOnnxPolicy(policy_root) if use_official_policy else None
        expected_joint_names = [*spec.policy_actuators, *spec.locked_actuators]
        simulation_context = SimulationContext(
            sim_utils.SimulationCfg(dt=contract["physics_dt_s"], device="cuda:0")
        )
        phase = "simulation_context_created"
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        robot = Articulation(make_isaac_articulation_cfg().replace(prim_path="/World/TRON2A"))
        simulation_context.reset()
        phase = "articulation_imported"
        runtime_joint_names = list(robot.data.joint_names)
        if set(runtime_joint_names) != set(expected_joint_names) or len(runtime_joint_names) != len(expected_joint_names):
            raise RuntimeError(
                f"runtime joints differ from phase-1 contract: {runtime_joint_names} != {expected_joint_names}"
            )
        policy_runtime_ids = runtime_joint_ids(runtime_joint_names, spec.policy_actuators)
        locked_ids = runtime_joint_ids(runtime_joint_names, spec.locked_actuators)
        targets_by_name = make_joint_position_targets(spec, policy_spec, [0.0] * policy_spec.policy_output_size)
        position_targets = robot.data.joint_pos.clone()
        for runtime_id, name in enumerate(runtime_joint_names):
            position_targets[0, runtime_id] = targets_by_name[name]
        robot.write_root_state_to_sim(robot.data.default_root_state)
        robot.write_joint_state_to_sim(position_targets, torch.zeros_like(position_targets))
        robot.set_joint_position_target(position_targets)
        robot.write_data_to_sim()
        locked_targets = torch.tensor(
            [spec.locked_actuators[name].target_position for name in spec.locked_actuators],
            dtype=torch.float32,
            device=simulation_context.device,
        )
        steps = int(contract["duration_s"] / contract["physics_dt_s"])
        min_base_height = float("inf")
        max_locked_error = 0.0
        max_locked_errors = {name: 0.0 for name in spec.locked_actuators}
        finite = True
        policy_action_count = 0
        captured_frame_count = 0
        last_actions = [0.0] * policy_spec.policy_output_size
        for step in range(steps):
            if policy_runner is not None and step % policy_spec.decimation == 0:
                gravity_w = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=simulation_context.device)
                projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity_w)[0]
                policy_joint_pos = robot.data.joint_pos[0, policy_runtime_ids]
                policy_joint_vel = robot.data.joint_vel[0, policy_runtime_ids]
                observation = torch.cat(
                    (
                        robot.data.root_ang_vel_b[0] * 0.25,
                        projected_gravity,
                        policy_joint_pos - torch.tensor(
                            [policy_spec.default_joint_positions[name] for name in policy_spec.action_joint_names],
                            dtype=torch.float32,
                            device=simulation_context.device,
                        ),
                        policy_joint_vel * 0.05,
                        torch.tensor(last_actions, dtype=torch.float32, device=simulation_context.device),
                        torch.tensor([0.0, 1.0, 0.8, 0.5, 0.5, 0.2], device=simulation_context.device),
                    )
                )
                last_actions = policy_runner.infer(observation.detach().cpu().numpy())
                targets_by_name = make_joint_position_targets(spec, policy_spec, last_actions)
                for runtime_id, name in enumerate(runtime_joint_names):
                    position_targets[0, runtime_id] = targets_by_name[name]
                robot.set_joint_position_target(position_targets)
                policy_action_count += 1
            robot.write_data_to_sim()
            simulation_context.step()
            robot.update(simulation_context.get_physics_dt())
            tensors = (robot.data.root_pos_w, robot.data.root_quat_w, robot.data.joint_pos, robot.data.joint_vel)
            finite = finite and all(bool(torch.isfinite(value).all()) for value in tensors)
            min_base_height = min(min_base_height, float(robot.data.root_pos_w[0, 2].item()))
            if cam_track:
                base_position = robot.data.root_pos_w[0]
                x, y, z = (float(value) for value in base_position.tolist())
                simulation_context.set_camera_view((x + 2.4, y - 2.4, max(z + 1.35, 1.35)), (x, y, z + 0.55))
            if frame_directory is not None and step % 50 == 0:
                capture_viewport_to_file(viewport, str(frame_directory / f"frame_{captured_frame_count:05d}.png"))
                captured_frame_count += 1
            locked_positions = robot.data.joint_pos[0, locked_ids]
            locked_errors = torch.abs(locked_positions - locked_targets)
            max_locked_error = max(max_locked_error, float(torch.max(locked_errors).item()))
            for name, error_value in zip(spec.locked_actuators, locked_errors.tolist(), strict=True):
                max_locked_errors[name] = max(max_locked_errors[name], float(error_value))

        checks = {
            "finite_state": finite,
            "locked_joint_hold": max_locked_error <= contract["maximum_locked_joint_error"],
        }
        if use_official_policy:
            checks["policy_base_height"] = min_base_height >= 0.55
        payload = {
            "artifact_schema_version": 1,
            "status": "passed" if all(checks.values()) else "failed",
            "headless": headless,
            "official_policy": use_official_policy,
            "official_policy_action_count": policy_action_count,
            "cam_track": cam_track,
            "record_mp4": str(record_mp4) if record_mp4 is not None else None,
            "recorded_frame_count": captured_frame_count,
            "duration_s": contract["duration_s"],
            "physics_dt_s": contract["physics_dt_s"],
            "steps": steps,
            "root_link": spec.root_link,
            "runtime_joint_names": runtime_joint_names,
            "policy_joint_runtime_ids": policy_runtime_ids,
            "locked_joint_runtime_ids": locked_ids,
            "minimum_base_height_m": min_base_height,
            "maximum_locked_joint_error": max_locked_error,
            "maximum_locked_joint_errors": max_locked_errors,
            "checks": checks,
        }
        diagnostic = payload
        _write_artifact(artifact_path, payload)
        if payload["status"] != "passed":
            raise RuntimeError(f"phase-2 runtime validation failed: {checks}")
        return payload
    except BaseException as error:
        failure = {
            "artifact_schema_version": 1,
            "status": "failed",
            "phase": phase,
            "error": f"{type(error).__name__}: {error}",
        }
        if diagnostic is not None:
            failure["diagnostic"] = diagnostic
        _write_artifact(artifact_path, failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        raise
    finally:
        if simulation_context is not None:
            simulation_context.clear_all_callbacks()
            from isaaclab.sim import SimulationContext

            SimulationContext.clear_instance()
        if frame_directory is not None and ffmpeg_binary is not None:
            frame_pattern = frame_directory / "frame_%05d.png"
            subprocess.run(
                [
                    str(ffmpeg_binary), "-y", "-framerate", "4", "-i", str(frame_pattern), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", str(record_mp4),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        app.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", action="store_true", help="Require Isaac Lab and construct the articulation config.")
    parser.add_argument("--headless", action="store_true", help="Run the Isaac validation without a viewport.")
    parser.add_argument("--official-policy", action="store_true", help="Run the pinned SF_TRON2A ONNX policy.")
    parser.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--duration-s", type=float, help="Override the default 10-second validation duration.")
    parser.add_argument("--cam-track", action="store_true", help="Follow base_Link with the GUI viewport camera.")
    parser.add_argument("--record-mp4", type=Path, help="Capture the GUI viewport desktop to an MP4 file.")
    parser.add_argument("--capture-size", default="1920x1080", help="Desktop recording size, for example 2560x1440.")
    parser.add_argument("--artifact", type=Path, default=RUNTIME_ARTIFACT_PATH)
    args = parser.parse_args()
    report = inspect_offline_training_asset()
    if args.isaac:
        report = run_isaac_runtime_validation(
            args.headless, args.artifact, args.official_policy, args.policy_root, args.duration_s,
            args.cam_track, args.record_mp4, args.capture_size,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
