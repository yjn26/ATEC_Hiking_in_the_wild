"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from play_controls import apply_keyboard_velocity_delta

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--useonnx", action="store_true", default=False, help="Use the exported ONNX model for inference.")
parser.add_argument(
    "--onnx_model_dir",
    type=str,
    default=None,
    help="Directory containing actor.onnx and 0-depth_encoder.onnx, or the actor.onnx file itself.",
)
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument("--keyboard_control", action="store_true", default=False, help="Enable keyboard control.")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="Linear velocity change per keyboard step.")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="Yaw-rate change per keyboard step.")
parser.add_argument("--no_depth_viewer", action="store_true", help="Do not open the Qt depth and velocity viewer.")
# cobblestone trough scene (built by pebble_loader.py, Isaac/PhysX only)
parser.add_argument(
    "--pebble_scene",
    type=str,
    default=None,
    help="Path to the pebble bundle dir (contains manifest.json + layouts.npz). Enables the 1.5x4 m trough demo.",
)
parser.add_argument("--pebble_layout", type=str, default="0", help="Layout id 0-15, or 'random'.")
parser.add_argument("--pebble_mode", type=str, default="dynamic", choices=["static", "dynamic"], help="Pebble physics mode.")
parser.add_argument("--pebble_hide", action="store_true", default=False, help="Hide pebble meshes (physics only, no visuals).")
parser.add_argument("--pebble_yaw_deg", type=float, default=180.0, help="Spawn yaw in degrees (180 = facing -X, straight at the trough).")
parser.add_argument(
    "--pebble_auto_forward",
    type=float,
    default=0.0,
    help="If >0, drive a constant forward speed instead of WASD (headless smoke test).",
)
parser.add_argument(
    "--pebble_contact_offset",
    type=float,
    default=0.0,
    help=(
        "PhysX contact offset for the pebbles (m). 0 = engine default (20 mm), which is the same "
        "size as a pebble and makes the whole bed behave like a rigid slab. Use 0.001-0.002 to see "
        "the bed yield. Note a contact pair uses the SUM of both offsets, so the robot colliders "
        "need the same treatment (--pebble_robot_contact_offset)."
    ),
)
parser.add_argument(
    "--pebble_rest_offset",
    type=float,
    default=0.0,
    help="PhysX rest offset for the pebbles (m). 0 = engine default. Half of contact offset is typical.",
)
parser.add_argument(
    "--pebble_robot_contact_offset",
    type=float,
    default=0.0,
    help="If >0, apply this contact offset to every collider of the robot (the other side of the contact pair).",
)
parser.add_argument(
    "--pebble_spawn_xy",
    type=str,
    default="",
    help="Override the spawn position on the pad, e.g. '0.0,-1.6' to start inside the trough.",
)
parser.add_argument(
    "--pebble_spawn_z",
    type=float,
    default=-1.0,
    help="Height of the surface the robot stands on (default: the standing pad top, 0.255 m).",
)
parser.add_argument(
    "--pebble_cam_eye",
    type=str,
    default="3.4,-4.6,2.3",
    help="Fixed world-space camera position for the demo, 'x,y,z'.",
)
parser.add_argument(
    "--pebble_cam_lookat",
    type=str,
    default="-0.3,-0.5,0.45",
    help="Fixed world-space camera target for the demo, 'x,y,z'.",
)

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import carb.input
from carb.input import KeyboardEventType
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components, get_subobs_size

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def main():
    """Play with Instinct-RL agent."""
    import instinctlab as _il
    import gymnasium as _gym
    print("[DBG] instinctlab =", _il.__file__, flush=True)
    print("[DBG] tasks =", [k for k in _gym.registry.keys() if "Instinct" in k][:8], flush=True)
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    direct_onnx = args_cli.onnx_model_dir is not None
    if direct_onnx:
        if args_cli.exportonnx:
            raise ValueError("--exportonnx cannot be used with --onnx_model_dir.")
        if args_cli.env_cfg or args_cli.agent_cfg:
            raise ValueError("--env_cfg and --agent_cfg require a PyTorch checkpoint run.")
        args_cli.useonnx = True
        from onnxer import validate_parkour_onnx_model_dir

        onnx_model_dir = validate_parkour_onnx_model_dir(args_cli.onnx_model_dir)
        print(f"[INFO]: Using direct ONNX policy from: {onnx_model_dir}")
        # Deployment ONNX exports have a fixed batch dimension of one.
        env_cfg.scene.num_envs = 1

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if direct_onnx:
        log_dir = str(onnx_model_dir.parent)
        resume_path = None
    elif agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        env_cfg = load_yaml(os.path.join(log_dir, "params", "env.yaml"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    if args_cli.keyboard_control:
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10

    pebble_asset = None
    if args_cli.pebble_scene:
        # The trough demo is a single-robot, single-env interactive scene.
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10
        args_cli.keyboard_control = True

        import pebble_scene_bridge as _pb

        # (a) terrain: the trough floor is modelled at z = 0 (`trough_floor` spans [-0.05, 0]),
        # so a plane terrain *is* the trough floor. The generated parkour terrain undulates by
        # up to 0.2 m (same size as the 0.20 m walls -> trough buried) and it moves env_origins
        # around on reset (-> robot and trough end up 16 m apart).
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        # the parkour cfg asks for an MDL ground material, which a plane cannot sample ->
        # IsaacLab falls back to pure black; use a plain grey preview surface instead.
        from isaaclab.sim import PreviewSurfaceCfg

        env_cfg.scene.terrain.visual_material = PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45))

        # (b) spawn: `init_state.pos` is the robot *root* pose, not the feet. Stand on the
        # 1.20x1.00 m pad by adding the default standing root height (0.9 m for the G1 cfg)
        # on top of the pad surface, facing the trough so W walks off the platform.
        _root_z = float(env_cfg.scene.robot.init_state.pos[2])
        _spawn_xy = None
        if args_cli.pebble_spawn_xy:
            _sx, _sy = (float(v) for v in args_cli.pebble_spawn_xy.split(","))
            _spawn_xy = (_sx, _sy)
        _spawn_pos, _spawn_rot = _pb.robot_spawn(
            args_cli.pebble_scene,
            _root_z,
            args_cli.pebble_yaw_deg,
            spawn_xy=_spawn_xy,
            surface_z=(args_cli.pebble_spawn_z if args_cli.pebble_spawn_z >= 0.0 else None),
        )
        env_cfg.scene.robot.init_state.pos = _spawn_pos
        env_cfg.scene.robot.init_state.rot = _spawn_rot
        print(
            f"[INFO] robot spawn on pad: pos=({_spawn_pos[0]:.3f}, {_spawn_pos[1]:.3f}, "
            f"{_spawn_pos[2]:.3f}) = pad_top + root {_root_z:.3f} m, yaw={args_cli.pebble_yaw_deg:.1f} deg"
        )

        # (c) pebbles: must be spawned from the SceneCfg so that they exist *before*
        # `sim.reset()`; creating rigid bodies afterwards makes the PhysX fabric views stale
        # and dynamic mode dies with `CUDA error: illegal memory access`.
        pebble_asset = _pb.make_scene_asset(
            args_cli.pebble_scene,
            layout_id=args_cli.pebble_layout,
            mode=args_cli.pebble_mode,
            hide=args_cli.pebble_hide,
            contact_offset=args_cli.pebble_contact_offset,
            rest_offset=args_cli.pebble_rest_offset,
            robot_prim_path=str(env_cfg.scene.robot.prim_path).format(ENV_REGEX_NS="/World/envs/env_0"),
            robot_contact_offset=args_cli.pebble_robot_contact_offset,
        )
        env_cfg.scene.pebble = pebble_asset

        # 相机从侧上方俯视，能看到整条 4 m 长槽和机器人把卵石推开的痕迹。
        # 固定机位，高度约 25 cm，同时看得清台面、槽壁和机器人的轮廓。
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = tuple(float(v) for v in args_cli.pebble_cam_eye.split(","))
        env_cfg.viewer.lookat = tuple(float(v) for v in args_cli.pebble_cam_lookat.split(","))

    model_label = Path(resume_path).stem.split("_")[-1] if resume_path is not None else onnx_model_dir.parent.name

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    pebble_info = None
    if pebble_asset is not None:
        import pebble_scene_bridge as _pb

        pebble_info = _pb.last_scene()
        if pebble_info is None:
            raise RuntimeError("pebble scene asset was registered but never spawned")
        print(f"[INFO] pebble scene spawned: {_pb.summary(pebble_info)}")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{model_label}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    policy = None
    if not direct_onnx:
        ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        if agent_cfg.load_run is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        if args_cli.sample:
            policy = ppo_runner.alg.actor_critic.act
        else:
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    # use the exported model for inference
    if args_cli.useonnx:
        from onnxer import load_parkour_onnx_model

        # NOTE: This is only applicable with parkour task
        onnx_policy = load_parkour_onnx_model(
            model_dir=onnx_model_dir if direct_onnx else os.path.join(log_dir, "exported"),
            get_subobs_func=lambda obs: get_subobs_by_components(
                obs,
                agent_cfg.policy.encoder_configs.depth_encoder.component_names,
                env.get_obs_segments(),
                temporal=True,
            ),
            depth_shape=env.get_obs_segments()["depth_image"],
            proprio_slice=slice(
                0,
                get_subobs_size(
                    env.get_obs_segments(),
                    [
                        "base_lin_vel",
                        "base_ang_vel",
                        "projected_gravity",
                        "velocity_commands",
                        "joint_pos",
                        "joint_vel",
                        "actions",
                    ],
                ),
            ),
        )
        print(f"[INFO]: Loaded parkour ONNX inference sessions from: {onnx_model_dir if direct_onnx else export_model_dir}")

    override_command = torch.zeros(env.num_envs, 3, device=env.device)
    if args_cli.pebble_auto_forward > 0.0:
        override_command[:, 0] = args_cli.pebble_auto_forward
        print(f"[INFO] pebble auto-forward: lin_vel_x = {args_cli.pebble_auto_forward:.2f} m/s")
    command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")
    dashboard = None
    if not args_cli.headless and not args_cli.no_depth_viewer:
        from play_dashboard import QtDepthDashboard

        dashboard = QtDepthDashboard()

    def update_dashboard():
        if dashboard is None:
            return
        camera_output = env.unwrapped.scene["camera"].data.output
        depth_key = "distance_to_image_plane_noised"
        if depth_key not in camera_output:
            depth_key = "distance_to_image_plane"
        depth_image = camera_output[depth_key][0, ..., 0].detach().cpu().numpy()
        if args_cli.keyboard_control:
            command = override_command[0]
        else:
            command = env.unwrapped.command_manager.get_command("base_velocity")[0]
        robot = env.unwrapped.scene["robot"]
        actual = torch.cat((robot.data.root_lin_vel_b[0, :2], robot.data.root_ang_vel_b[0, 2:3]))
        dashboard.update(depth_image, command.detach().cpu().numpy(), actual.detach().cpu().numpy())

    def on_keyboard_input(e):
        if e.type not in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
            return
        if e.input == carb.input.KeyboardInput.W:
            override_command.copy_(apply_keyboard_velocity_delta(override_command, linear_delta=args_cli.keyboard_linvel_step))
        elif e.input == carb.input.KeyboardInput.S:
            override_command.copy_(apply_keyboard_velocity_delta(override_command, linear_delta=-args_cli.keyboard_linvel_step))
        elif e.input == carb.input.KeyboardInput.A:
            override_command.copy_(apply_keyboard_velocity_delta(override_command, yaw_delta=args_cli.keyboard_angvel))
        elif e.input == carb.input.KeyboardInput.D:
            override_command.copy_(apply_keyboard_velocity_delta(override_command, yaw_delta=-args_cli.keyboard_angvel))
        elif e.input == carb.input.KeyboardInput.X:
            override_command.zero_()

    # NOTE: the default app window does not exist in --headless mode, so WASD is GUI-only.
    if not args_cli.headless:
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        input = carb.input.acquire_input_interface()
        input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    # reset environment
    _pebble_before = None
    obs, _ = env.get_observations()
    if pebble_info is not None:
        import pebble_scene_bridge as _pb

        # With a plane terrain env_origins is identically zero, so the field needs no
        # translation at all. Anything else means the terrain was not replaced.
        _o = env.unwrapped.scene.terrain.env_origins[0]
        print(
            f"[INFO] terrain origin after reset: ({float(_o[0]):.3f}, {float(_o[1]):.3f}, {float(_o[2]):.3f})",
            flush=True,
        )
        _pb.realign_field(pebble_info, _o)
        _pebble_before = _pb.pebble_positions(pebble_info) if args_cli.pebble_mode == "dynamic" else None
        if _pebble_before is not None:
            print("[INFO] captured initial pebble bed positions (dynamic mode)", flush=True)
    _peb_traj = []      # robot base XY every 10 steps, exported with the pebble field
    if pebble_info is not None and args_cli.pebble_auto_forward > 0.0:
        print("[INFO] pebble smoke: tracking robot base position every 100 steps")
    update_dashboard()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.keyboard_control:
                obs[:, command_obs_slice[0]] = override_command.repeat(1, command_obs_slice[1][0] // 3)
            if args_cli.useonnx:
                actions = onnx_policy(obs)
                if policy is not None:
                    torch_actions = policy(obs)
                if policy is not None and (actions - torch_actions).abs().max() > 1e-5:
                    print(
                        "[INFO]: ONNX model and PyTorch model have a difference of"
                        f" {(actions - torch_actions).abs().max()} in actions at joint"
                        f" {((actions - torch_actions).abs() > 1e-5).nonzero(as_tuple=True)[0]}"
                    )
            else:
                actions = policy(obs)
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
        timestep += 1
        if _pebble_before is not None and timestep == 30:
            # 前 30 步还没下台面：这段位移是卵石床自沉降，不是机器人压出来的，
            # 单独报一次后把基线换成沉降后的状态。
            _pb.report_displacement(pebble_info, _pebble_before, label="settling(0-30 steps)")
            _pebble_before = _pb.pebble_positions(pebble_info)
        if pebble_info is not None and args_cli.pebble_auto_forward > 0.0 and timestep % 100 == 0:
            robot = env.unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w[0]
            vel_b = robot.data.root_lin_vel_b[0]
            cmd = override_command[0]
            print(
                f"[pebble] step={timestep} base=({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}) "
                f"vel_b=({vel_b[0]:.2f}, {vel_b[1]:.2f}) cmd=({cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f})",
                flush=True,
            )
        if _peb_traj is not None and timestep % 10 == 0:
            _rp = env.unwrapped.scene["robot"].data.root_pos_w[0]
            _peb_traj.append((float(_rp[0]), float(_rp[1])))
        if timestep % 5 == 0:
            update_dashboard()

        # exit the loop if video_length is meet
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    if pebble_info is not None and _pebble_before is not None:
        _pb.report_displacement(pebble_info, _pebble_before, label="final")
        try:
            _pb.export_field(
                pebble_info,
                _pebble_before,
                os.path.join(log_dir, "pebble_field_final.npz"),
                traj=_peb_traj,
            )
        except Exception as _exc:  # noqa: BLE001  (never break a demo over diagnostics)
            print("[INFO] pebble field export skipped: %r" % (_exc,), flush=True)
    env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{model_label}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
