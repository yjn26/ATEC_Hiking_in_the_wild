from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tracking_exp_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
    log_per_terrain: bool = True,
) -> dict:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
    (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        asset_cfg: The configuration of the robot articulation in the scene.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
            score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
            If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
            score (exponential kernel).
            Similar logic applies as lin_vel_threshold.
    Returns:
        A dict with the mean terrain level over all environments under ``"terrain_levels"`` plus one entry
        per sub-terrain type (e.g. ``"pebble_bed"``) when ``log_per_terrain`` is set.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    logged: dict[str, torch.Tensor] = {"terrain_levels": torch.mean(levels)}
    # 按地形类型分别记账，方便在 TensorBoard 里看某一种地形（例如 pebble_bed）推进到第几级
    if log_per_terrain:
        generator_cfg = getattr(terrain.cfg, "terrain_generator", None)
        if generator_cfg is not None and generator_cfg.sub_terrains:
            names = list(generator_cfg.sub_terrains.keys())
            proportions = np.array([sub.proportion for sub in generator_cfg.sub_terrains.values()], dtype=np.float64)
            proportions /= np.sum(proportions)
            cumulative = np.cumsum(proportions)
            # 复刻 TerrainGenerator / PoseVelocityCommand 的列 -> 地形类型映射
            column_type = np.array(
                [
                    int(np.min(np.where(index / generator_cfg.num_cols + 0.001 < cumulative)[0]))
                    for index in range(generator_cfg.num_cols)
                ],
                dtype=np.int32,
            )
            env_column = terrain.terrain_types.detach().cpu().numpy().astype(np.int32)
            env_type = column_type[np.clip(env_column, 0, generator_cfg.num_cols - 1)]
            for type_index, name in enumerate(names):
                mask = torch.from_numpy(env_type == type_index).to(levels.device)
                if bool(mask.any()):
                    logged[name] = torch.mean(levels[mask])
    return logged
