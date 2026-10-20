# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-only exporter for the parkour curriculum.

This script bakes the terrain **exactly the way the training run does** and writes
it out as OBJ / PNG / CSV.  It deliberately calls
:class:`isaaclab.terrains.TerrainGenerator` instead of the raw sub-terrain
callables.

Why that matters: ``TerrainGenerator.__init__`` overrides, for every sub-terrain
config,

* ``size``                 -> ``cfg.size``               (8.0, 8.0) m  (not the 10 m cfg default)
* ``horizontal_scale``     -> ``cfg.horizontal_scale``   0.05 m        (not the 0.1 m default)
* ``vertical_scale``       -> ``cfg.vertical_scale``     0.005 m
* ``slope_threshold``      -> ``cfg.slope_threshold``    1.0

and the curriculum difficulty of row ``r`` is ``(r + U(0,1)) / num_rows``, not
a plain ``linspace(0, 1, 10)``.  Calling ``cfg.function(difficulty, cfg)`` by
hand therefore produces a 10 m tile on a 0.1 m grid — the wrong terrain.

Terrain-only means: no environment, no robot, no policy, no PPO.  Kit is started
only because the config module imports Isaac Lab asset/sensor configs.

Outputs (``--out``)::

    obj/<terrain>_L<row>.obj   one tile per (sub-terrain, curriculum row)
    png/<terrain>.png          hill-shaded height map of the 10 rows
    png/overview.png           the whole 80 m x 160 m field, labelled by terrain
    stats.csv                  per-tile height / slope / roughness statistics
    meta.json                  generator parameters + column layout

NOTE on the two intentional preview overrides:

* ``wall_prob = 0``      — the curriculum puts 5 m guard walls on every tile edge
  (``wall_prob=[0.3, 0.3, 0.3, 0.3]``); they are pure border geometry (they do not modify
  the height field) and they ruin both the OBJ and the plots.  The exported heights are
  bit-identical to training.
* ``flat_patch_sampling = None`` — sampling spawn patches does not touch the
  mesh; dropping it takes the bake from ~22 s to ~3 s.

Examples::

    python scripts/pebble_terrain_preview.py --headless --out /tmp/terrain_preview
    python scripts/pebble_terrain_preview.py --headless --terrains pebble_bed --no-obj
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export every parkour sub-terrain to OBJ + PNG.")
parser.add_argument("--out", type=str, default="/tmp/terrain_preview", help="output directory")
parser.add_argument("--terrains", type=str, default="", help="comma separated subset, default: all")
parser.add_argument("--rows", type=int, default=10, help="number of curriculum difficulty rows")
parser.add_argument("--no-obj", action="store_true", help="skip the (large) OBJ export")
parser.add_argument("--no-png", action="store_true", help="skip the PNG rendering")
parser.add_argument("--vexag", type=float, default=1.0, help="hill-shade vertical exaggeration")
parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# the imports below need the Kit app (they pull pxr)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402

from isaaclab.terrains import TerrainGenerator  # noqa: E402

from instinctlab.tasks.parkour.config.parkour_env_cfg import ROUGH_TERRAINS_CFG  # noqa: E402


# --------------------------------------------------------------------------------------
# generator helpers
# --------------------------------------------------------------------------------------
def build_generator(cfg, rows: int):
    """Bake the curriculum without the 5 m guard walls and without spawn-patch sampling."""
    cfg = copy.deepcopy(cfg)
    cfg.num_rows = rows
    for sub_cfg in cfg.sub_terrains.values():
        # terrain-only export: the guard walls are not part of the height field and would
        # dominate every OBJ / plot.  Zero the probability *and* the height: leaving
        # wall_height=0 with wall_prob > 0 emits a row of zero-height faces (a degenerate
        # 5 cm skirt that shows up as spurious 90 deg faces and x_span = 8.05 m).
        if getattr(sub_cfg, "wall_prob", None):
            sub_cfg.wall_prob = [0.0] * len(sub_cfg.wall_prob)
        if hasattr(sub_cfg, "wall_height"):
            sub_cfg.wall_height = 0.0
        if getattr(sub_cfg, "flat_patch_sampling", None) is not None:
            sub_cfg.flat_patch_sampling = None
    gen = TerrainGenerator(cfg)
    return gen


def column_layout(cfg):
    """Reproduce ``_generate_curriculum_terrains``' column -> sub-terrain mapping."""
    proportions = np.array([sub_cfg.proportion for sub_cfg in cfg.sub_terrains.values()])
    proportions /= np.sum(proportions)
    names = list(cfg.sub_terrains.keys())
    cols = []
    for index in range(cfg.num_cols):
        sub_index = int(np.min(np.where(index / cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))
        cols.append(names[sub_index])
    return cols


def tile_mesh(gen, row: int, col: int):
    """Copy of one baked tile, re-centred on its own local frame.

    ``TerrainGenerator`` stores the tiles in ``col-major`` order (see
    ``_generate_curriculum_terrains``); the border mesh is appended *after* them,
    so it never shifts these indices.
    """
    index = col * gen.cfg.num_rows + row
    mesh = gen.terrain_meshes[index].copy()
    # ``_get_terrain_mesh`` centres the tile on (0, 0) and ``_add_sub_terrain`` shifts it by
    # (row + 0.5) * size, so the tile ends up occupying row*size .. (row+1)*size of the field
    # (row 0 == the easiest curriculum row).  Removing just that row/col offset gives the tile
    # in its own 0..size frame, which is what the exporters and the height-map grid expect.
    # (Adding or subtracting another half tile silently empties 3/4 of the grid.)
    mesh.apply_translation(np.array([-row * gen.cfg.size[0], -col * gen.cfg.size[1], 0.0]))
    return mesh


def heightmap(mesh, size, hs):
    """Re-sample a regular-grid tile mesh back onto its (n+1, n+1) grid."""
    n = int(round(size[0] / hs))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    ix = np.rint(verts[:, 0] / hs).astype(np.int64)
    iy = np.rint(verts[:, 1] / hs).astype(np.int64)
    ok = (ix >= 0) & (ix <= n) & (iy >= 0) & (iy <= n)
    h = np.full((n + 1, n + 1), np.nan)
    h[ix[ok], iy[ok]] = verts[ok, 2]  # index 0 == x, index 1 == y
    return h


def mesh_stats(mesh) -> dict:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    z = verts[:, 2]
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1)
    good = nn > 1e-12  # degenerate (zero-area) faces have no slope; keep them out of the stats
    nz = np.abs(n[good, 2]) / nn[good]
    slope = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))
    area = 0.5 * nn[good]
    above = float(area[z[faces[good]].mean(axis=1) > 0.005].sum() / max(area.sum(), 1e-12))
    return {
        "n_vert": int(len(verts)),
        "n_tri": int(len(faces)),
        "n_degenerate": int((~good).sum()),
        "x_span": float(verts[:, 0].max() - verts[:, 0].min()),
        "y_span": float(verts[:, 1].max() - verts[:, 1].min()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_mean": float(z.mean()),
        "z_p95": float(np.percentile(z, 95)),
        "slope_mean": float(slope.mean()),
        "slope_p95": float(np.percentile(slope, 95)),
        "slope_p999": float(np.percentile(slope, 99.9)),
        "slope_max": float(slope.max()),
        "area_above_5mm": above,
    }


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def hillshade(h, hs, vexag):
    ls = LightSource(azdeg=315, altdeg=45)
    hf = np.nan_to_num(h, nan=0.0)
    rgb = ls.shade(hf, cmap=plt.get_cmap("YlOrBr"), blend_mode="overlay",
                   vert_exag=vexag, dx=hs, dy=hs)
    return rgb


def render_terrain(panels, name, hs, vexag, dpi, path):
    """Contact sheet: one hill-shaded height map per curriculum row."""
    n = len(panels)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.3 * nrow), squeeze=False)
    zmax = max(float(np.nanmax(h)) for _, h in panels)
    zmin = min(float(np.nanmin(h)) for _, h in panels)
    for i, (title, h) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(hillshade(h, hs, vexag), origin="lower",
                  extent=(0, h.shape[1] * hs, 0, h.shape[0] * hs))
        ax.set_title("%s  z=[%.3f, %.3f] m" % (title, np.nanmin(h), np.nanmax(h)), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("y [m]", fontsize=6)
        ax.set_ylabel("x [m]", fontsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # inset colour bar taken from a plain height map of the first panel
    sm = plt.cm.ScalarMappable(cmap="YlOrBr", norm=plt.Normalize(zmin, max(zmax, zmin + 1e-6)))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("z [m]", fontsize=8)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("%s — sub-terrain tile, 10 curriculum rows (row 0 easiest)" % name, fontsize=11)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def field_heightmap(gen, cols, rows, size, hs):
    """Height map of the whole field, assembled from every baked tile.

    The field is ``rows * size[0]`` (x) by ``len(cols) * size[1]`` (y); tile (row, col)
    occupies ``[row*size[0], (row+1)*size[0]) x [col*size[1], (col+1)*size[1])``.  The
    closing row/column of each tile is dropped so tiles abut without overlapping.
    """
    n = int(round(size[0] / hs))
    h_field = np.full((rows * n, len(cols) * n), np.nan)
    for col, name in enumerate(cols):
        for row in range(rows):
            h = heightmap(tile_mesh(gen, row, col), size, hs)
            h_field[row * n:(row + 1) * n, col * n:(col + 1) * n] = h[:n, :n]
    return h_field


def render_overview(h, labels, per_class, size, hs, vexag, dpi, path, inset=None):
    """Whole field, grey hill-shade so that 3 cm cobbles stay visible next to 1.3 m pyramids."""
    nr, nc = int(round(h.shape[0] * hs / size[0])), int(round(h.shape[1] * hs / size[1]))
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(np.nan_to_num(h, nan=0.0), cmap=plt.get_cmap("gray"), blend_mode="overlay",
                   vert_exag=vexag, dx=hs, dy=hs)
    ax.imshow(rgb, origin="lower", extent=(0, nc * size[1], 0, nr * size[0]))
    for c in range(nc + 1):
        ax.axvline(c * size[1], color="k", lw=0.3, alpha=0.30)
    for r in range(nr + 1):
        ax.axhline(r * size[0], color="k", lw=0.3, alpha=0.30)
    seen = {}
    for c, name in enumerate(labels):
        seen.setdefault(name, []).append(c)
    for name, col_idx in seen.items():
        if name not in per_class:  # --terrains subset: column was not baked
            continue
        stats = per_class[name]
        zlo = min((s["z_min"] for s in stats), default=float("nan"))
        zhi = max((s["z_max"] for s in stats), default=float("nan"))
        c0, c1 = col_idx[0], col_idx[-1] + 1
        ax.text(0.5 * (c0 + c1) * size[1], nr * size[0] * 0.5,
                "%s  x%d\nz=[%.3f, %.3f] m" % (name.replace("_", " "), len(col_idx), zlo, zhi),
                rotation=90, ha="center", va="center", fontsize=8, color="k",
                bbox=dict(fc="w", ec="none", alpha=0.80, pad=1.5))
        for c in col_idx[1:]:
            ax.axvline(c * size[1], color="r", lw=0.8, ls="--", alpha=0.55)
    ax.set_xlabel("y [m]   num_cols = %d, one column per tile (%.0f m)" % (nc, size[1]), fontsize=9)
    ax.set_ylabel("x [m]   num_rows = %d, difficulty increases along x" % nr, fontsize=9)
    ax.set_title("parkour curriculum field  %.0f m x %.0f m   tile=%.0f m  grid=%.3f m  "
                 "z=[%.2f, %.2f] m" % (nr * size[0], nc * size[1], size[0], hs,
                                       np.nanmin(h), np.nanmax(h)), fontsize=10)
    if inset is not None:
        crop, title = inset
        iax = ax.inset_axes([0.765, 0.045, 0.085, 0.22])
        ls2 = LightSource(azdeg=315, altdeg=45)
        iax.imshow(hillshade(crop, hs, 4.0), origin="lower",
                   extent=(0, crop.shape[1] * hs, 0, crop.shape[0] * hs))
        iax.set_xticks([])
        iax.set_yticks([])
        iax.set_title(title, fontsize=6.5)
        for sp in iax.spines.values():
            sp.set_edgecolor("w")
            sp.set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    out = args_cli.out
    os.makedirs(os.path.join(out, "obj"), exist_ok=True)
    os.makedirs(os.path.join(out, "png"), exist_ok=True)

    raw = copy.deepcopy(ROUGH_TERRAINS_CFG)
    gen_cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    size = tuple(float(v) for v in raw.size)
    hs = float(raw.horizontal_scale)
    vs = float(raw.vertical_scale)
    print("[preview] generator: size=%s  horizontal_scale=%.3f  vertical_scale=%.4f  "
          "slope_threshold=%s  rows=%d  cols=%d"
          % (size, hs, vs, raw.slope_threshold, raw.num_rows, raw.num_cols))

    # the pebble generator draws from the legacy global RNG (it is called once per tile and
    # TerrainGenerator does not forward its per-tile seed), so seed it to make reruns identical
    np.random.seed(int(raw.seed) if raw.seed is not None else 0)

    t0 = time.perf_counter()
    gen = build_generator(gen_cfg, args_cli.rows)
    bake_s = time.perf_counter() - t0
    print("[preview] baked %d tiles in %.2f s" % (len(gen.terrain_meshes), bake_s))

    cols = column_layout(gen.cfg)
    wanted = [t for t in args_cli.terrains.split(",") if t]
    names = [n for n in gen.cfg.sub_terrains.keys() if not wanted or n in wanted]
    print("[preview] column layout:", {n: cols.count(n) for n in gen.cfg.sub_terrains})

    csv_path = os.path.join(out, "stats.csv")
    with open(csv_path, "w", newline="\n") as fh:
        fh.write("terrain,row,nominal_difficulty,col_index,n_tri,n_vert,n_degenerate,x_span,y_span,"
                 "z_min,z_max,z_mean,z_p95,slope_mean,slope_p95,slope_p999,slope_max,area_above_5mm\n")

    grid_rows = gen.cfg.num_rows

    all_stats = {}
    for name in names:
        my_cols = [i for i, n in enumerate(cols) if n == name]
        col = my_cols[0]  # representative column: OBJ / PNG / stats export
        panels = []
        rows_stats = []
        t1 = time.perf_counter()
        for row in range(grid_rows):
            mesh = tile_mesh(gen, row, col)
            st = mesh_stats(mesh)
            rows_stats.append(st)
            nominal = (row + 0.5) / grid_rows
            h = heightmap(mesh, size, hs)
            if row == 0:
                cov = float(np.isfinite(h).mean())
                print("[preview] %-22s height-map coverage %.1f%% (expect ~100%%)%s"
                      % (name, 100.0 * cov,
                         "  <-- WARNING: tile frame is off" if cov < 0.98 else ""))
            panels.append(("L%02d  d~%.2f" % (row, nominal), h))
            if not args_cli.no_obj:
                mesh.export(os.path.join(out, "obj", "%s_L%02d.obj" % (name, row)))
            print("[preview] %-22s L%02d d=%.2f  tri=%7d (degen %3d)  xyz=(%.2f,%.2f)  "
                  "z=[%.3f, %.3f]  slope p95/p99.9/max=%5.1f/%5.1f/%5.1f"
                  % (name, row, nominal, st["n_tri"], st["n_degenerate"], st["x_span"], st["y_span"],
                     st["z_min"], st["z_max"], st["slope_p95"], st["slope_p999"], st["slope_max"]))
            with open(csv_path, "a", newline="\n") as fh:
                fh.write("%s,%d,%.3f,%d,%d,%d,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.4f\n"
                         % (name, row, nominal, col, st["n_tri"], st["n_vert"], st["n_degenerate"],
                            st["x_span"], st["y_span"], st["z_min"], st["z_max"], st["z_mean"],
                            st["z_p95"], st["slope_mean"], st["slope_p95"], st["slope_p999"],
                            st["slope_max"], st["area_above_5mm"]))
        all_stats[name] = rows_stats
        if not args_cli.no_png:
            png = os.path.join(out, "png", "%s.png" % name)
            render_terrain(panels, name, hs, args_cli.vexag, args_cli.dpi, png)
            print("[preview] %-22s wrote %s in %.1f s" % (name, png, time.perf_counter() - t1))

    if not args_cli.no_png:
        t2 = time.perf_counter()
        h_field = field_heightmap(gen, cols, grid_rows, size, hs)
        inset = None
        if "pebble_bed" in names and "pebble_bed" in cols:
            # cobbles are ~0.16 m stones: sub-pixel in the full-field view, so zoom 4 m
            hc = heightmap(tile_mesh(gen, grid_rows // 2, cols.index("pebble_bed")), size, hs)
            k = int(round(4.0 / hs))
            inset = (hc[:k, :k], "pebble_bed L%02d, 4 m crop" % (grid_rows // 2))
        ov = os.path.join(out, "png", "overview.png")
        render_overview(h_field, cols, all_stats, size, hs, args_cli.vexag, args_cli.dpi, ov,
                        inset=inset)
        print("[preview] wrote %s (%dx%d) in %.1f s"
              % (ov, h_field.shape[0], h_field.shape[1], time.perf_counter() - t2))

    meta = {
        "generator": {
            "size": list(size),
            "horizontal_scale": hs,
            "vertical_scale": vs,
            "slope_threshold": raw.slope_threshold,
            "num_rows": grid_rows,
            "num_cols": int(gen.cfg.num_cols),
            "difficulty_range": list(raw.difficulty_range),
            "seed": raw.seed,
        },
        "columns": cols,
        "column_counts": {n: cols.count(n) for n in gen.cfg.sub_terrains},
        "proportions": {n: float(c.proportion) for n, c in gen.cfg.sub_terrains.items()},
        "preview_overrides": {"wall_prob": 0.0, "flat_patch_sampling": None},
        "bake_seconds": round(bake_s, 3),
        "stats": all_stats,
    }
    with open(os.path.join(out, "meta.json"), "w", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("[preview] bake %.2f s total, done -> %s" % (bake_s, out))


if __name__ == "__main__":
    main()
    simulation_app.close()
