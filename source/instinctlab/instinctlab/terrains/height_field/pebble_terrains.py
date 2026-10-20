# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cobblestone / pebble bed terrain, generated procedurally on the height-field grid.

Why a height field instead of rigid pebbles
-------------------------------------------
The 4-layer bed shipped in ``scripts/pebble_scene`` (18 847 rigid stones, 0.12 m thick)
only makes sense in a *play* scene: with 4096 parallel environments the same bed would need
~77 M rigid bodies.  The terrain importer, on the other hand, builds **one static trimesh per
tile**, so every pebble-bed tile is a baked height field.  Consequence, stated plainly:
feet cannot push stones aside here -- this terrain teaches "walk on an irregular, locally
yielding-looking cobble surface", not the full granular dynamics.

Resolution
----------
The grid spacing is ``cfg.horizontal_scale`` (5 cm for the parkour task), so the stones are
generated bigger than the 2-6 cm gravel of the rigid bed: 12-26 cm cobbles standing 3-7 cm
proud, which is what a 5 cm lattice can actually represent.  Slopes are clamped so that no
cell exceeds ``max_slope_deg``; that keeps the mesh free of the vertical faces that the
importer's ``slope_threshold`` correction would otherwise create (and which the edge-cylinder
virtual obstacles would then flood with cylinders).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh

from .utils import generate_wall

if TYPE_CHECKING:
    from . import hf_terrains_cfg


def _rasterise_stone(
    heights: np.ndarray,
    cx: float,
    cy: float,
    semi_a: float,
    semi_b: float,
    yaw: float,
    crest: float,
    base_height: float,
    horizontal_scale: float,
) -> None:
    """Blend one ellipsoidal cobble dome into ``heights`` (in place, max-combine)."""
    n_x, n_y = heights.shape
    # bounding box of the ellipse (axis aligned, conservative)
    reach = max(semi_a, semi_b)
    i0 = max(0, int(np.floor((cx - reach) / horizontal_scale)))
    i1 = min(n_x, int(np.ceil((cx + reach) / horizontal_scale)) + 1)
    j0 = max(0, int(np.floor((cy - reach) / horizontal_scale)))
    j1 = min(n_y, int(np.ceil((cy + reach) / horizontal_scale)) + 1)
    if i1 <= i0 or j1 <= j0:
        return

    xs = (np.arange(i0, i1) + 0.5) * horizontal_scale - cx
    ys = (np.arange(j0, j1) + 0.5) * horizontal_scale - cy
    dx, dy = np.meshgrid(xs, ys, indexing="ij")

    cos_t, sin_t = np.cos(yaw), np.sin(yaw)
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t
    r2 = (u / semi_a) ** 2 + (v / semi_b) ** 2

    inside = r2 < 1.0
    if not np.any(inside):
        return
    # slightly flattened dome: r == 1 -> crest height zero, r == 0 -> crest
    dome = base_height + crest * (1.0 - r2) ** 1.0
    patch = heights[i0:i1, j0:j1]
    patch[inside] = np.maximum(patch[inside], dome[inside])


def _limit_slope(heights: np.ndarray, max_step: float, iterations: int) -> np.ndarray:
    """Erode peaks until no cell-to-cell step exceeds ``max_step`` (metres)."""
    h = heights.copy()
    for _ in range(max(0, iterations)):
        for axis in (0, 1):
            n = h.shape[axis]
            if n < 2:
                continue
            lo = np.roll(h, 1, axis=axis)
            hi = np.roll(h, -1, axis=axis)
            # 边缘格没有外侧邻居：把自己当邻居（不侵蚀边界，避免 -inf 传染）
            if axis == 0:
                lo[0, :] = h[0, :]
                hi[-1, :] = h[-1, :]
            else:
                lo[:, 0] = h[:, 0]
                hi[:, -1] = h[:, -1]
            h = np.minimum(h, np.minimum(lo, hi) + max_step)
    return h


def _blur(heights: np.ndarray, weight: float) -> np.ndarray:
    """3x3 box blur, mixed with the original by ``weight``."""
    if weight <= 0.0:
        return heights
    h = heights
    padded = np.pad(h, 1, mode="edge")
    acc = np.zeros_like(h)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            acc += padded[1 + di : 1 + di + h.shape[0], 1 + dj : 1 + dj + h.shape[1]]
    return (1.0 - weight) * h + weight * (acc / 9.0)


def pebble_bed_heights(difficulty: float, cfg: hf_terrains_cfg.PebbleBedTerrainCfg) -> np.ndarray:
    """Generate the cobble bed as a height array in **grid units** (int16)."""
    horizontal_scale = cfg.horizontal_scale
    n_x = int(cfg.size[0] / horizontal_scale)
    n_y = int(cfg.size[1] / horizontal_scale)
    if n_x < 2 or n_y < 2:
        raise ValueError("pebble bed terrain is too small for the given horizontal_scale.")

    # non-deterministic rng per tile (terrain generation is cached only if cfg.use_cache)
    rng = np.random.default_rng(int(np.random.randint(0, np.iinfo(np.int32).max)))

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    size_gain = 1.0 + cfg.size_difficulty_gain * difficulty

    r_lo = cfg.stone_radius_range[0] * size_gain
    r_hi = cfg.stone_radius_range[1] * size_gain
    r_mean = 0.5 * (r_lo + r_hi)

    # mean proud height of a stone, also grows with difficulty
    crest_lo = cfg.crest_height_ratio[0] * r_mean
    crest_hi = cfg.crest_height_ratio[1] * r_mean

    heights = np.zeros((n_x, n_y), dtype=np.float64)
    size_x = n_x * horizontal_scale
    size_y = n_y * horizontal_scale

    for layer in range(max(1, cfg.num_layers)):
        layer_scale = 1.0 if layer == 0 else cfg.upper_layer_scale
        lattice = 2.0 * r_mean * cfg.lattice_spacing_ratio * (1.0 if layer == 0 else 0.82)
        lattice = max(lattice, 1.5 * horizontal_scale)
        offset = 0.0 if layer == 0 else 0.5

        x_start = -r_mean + offset * lattice
        y_start = -r_mean + offset * lattice
        n_stones_x = int(np.ceil((size_x + 2.0 * r_mean) / lattice)) + 1
        n_stones_y = int(np.ceil((size_y + 2.0 * r_mean) / lattice)) + 1

        for ix in range(n_stones_x):
            for iy in range(n_stones_y):
                cx = x_start + ix * lattice + rng.uniform(-0.3, 0.3) * lattice
                cy = y_start + iy * lattice + rng.uniform(-0.3, 0.3) * lattice
                if cx < -r_hi or cx > size_x + r_hi or cy < -r_hi or cy > size_y + r_hi:
                    continue
                semi_a = rng.uniform(r_lo, r_hi) * layer_scale
                semi_b = semi_a * rng.uniform(0.72, 1.0)
                yaw = rng.uniform(0.0, np.pi)
                crest = rng.uniform(crest_lo, crest_hi) * layer_scale
                if layer == 0:
                    base_height = 0.0
                else:
                    # upper stones sit in the valleys of the layer below and poke above it
                    ci = int(np.clip(cx / horizontal_scale, 0, n_x - 1))
                    cj = int(np.clip(cy / horizontal_scale, 0, n_y - 1))
                    base_height = float(heights[ci, cj])
                _rasterise_stone(
                    heights, cx, cy, semi_a, semi_b, yaw, crest, base_height, horizontal_scale
                )

    # packed fines between the cobbles: the bed never opens down to the bare plate
    if cfg.crevice_fill_ratio > 0.0:
        heights = np.maximum(heights, cfg.crevice_fill_ratio * 0.5 * (crest_lo + crest_hi))

    # keep the mesh smooth enough that (a) the importer never turns a face vertical and
    # (b) the edge-cylinder virtual obstacles do not fire on every crevice
    max_step = np.tan(np.deg2rad(cfg.max_slope_deg)) * horizontal_scale
    heights = _limit_slope(heights, max_step, cfg.slope_iterations)
    heights = _blur(heights, cfg.blur_weight)
    heights = _limit_slope(heights, max_step, max(1, cfg.slope_iterations // 2))
    heights = np.clip(heights, 0.0, None)

    return (heights / cfg.vertical_scale).astype(np.int16)


@generate_wall
@height_field_to_mesh
def pebble_bed_terrain(difficulty: float, cfg: hf_terrains_cfg.PebbleBedTerrainCfg) -> np.ndarray:
    """Cobblestone bed terrain, see :func:`pebble_bed_heights`."""
    return pebble_bed_heights(difficulty, cfg)  # type: ignore[return-value, arg-type]
