#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_pebbles.py
==============
生成「鹅卵石步行槽 + 抬升平台 + 站立台」场景包（bundle），与仿真引擎无关。

场景约定（默认参数，全部来自 config JSON）
------------------------------------------
坐标: 右手系, Z 向上, 单位 m, 四元数 (w, x, y, z)
  槽体内腔: X in [-0.75, 0.75], Y in [-2.0, 2.0]   (1.5 m x 4.0 m)
  槽体深度: 0.20 m  (z_floor = 0, z_rim = 0.20)
  卵石层顶: 0.15 m nominal (最大卵石顶端 = 0.15)
  卵石下方填充体顶面: fill_z = 0.15 - 2*max(z_ext)  (~0.107)
  平台:     X in [0.80, 1.80], 顶面 z = 0.25  (高于卵石顶约 10 cm)
  站立台:   平台上 0.8 x 0.8 m 的薄板, 默认中心 (1.30, -1.55, +0.005)

输出 bundle
-----------
  manifest.json
  layout_config.json          实际生效配置（含派生量）
  shapes/rock_pool.npz        单位凸包池: verts / faces / offsets / volume
  shapes/rock_pool.obj        同上（可视化/外部工具用）
  layouts.npz                 每布局: shape_id / pos / quat(wxyz) / scale / mass / mu
  static/stage_boxes.json     槽/墙/平台/站立台 的 box 列表（推荐直接建 box 碰撞体）
  static/stage.obj            同上网格版（渲染用）
  static/cobble_L{ii}.obj     合并卵石表面网格（默认只导 layout 0，渲染用）
  static/heightfield_L{ii}.npy 每布局高度场 (float32, [nx, ny], i->X, j->Y)
  static/heightfield_meta.json
  preview.png

依赖: numpy, scipy  (matplotlib 仅用于 preview，可关)

用法
----
  python gen_pebbles.py --config config_1p5x4.json --out out_full
  python gen_pebbles.py --config config_smoke.json  --out out_smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import hashlib

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------

BOX_QUADS = [
    (0, 1, 3, 2),  # z-
    (4, 6, 7, 5),  # z+
    (0, 4, 5, 1),  # y-
    (2, 3, 7, 6),  # y+
    (0, 2, 6, 4),  # x-
    (1, 5, 7, 3),  # x+
]


def box_verts(center, size):
    """8 个顶点，顺序 (sz, sy, sx) 二进制的低位为 x。"""
    c = np.asarray(center, float)
    h = np.asarray(size, float) / 2.0
    out = []
    for sz in (-1, 1):
        for sy in (-1, 1):
            for sx in (-1, 1):
                out.append([c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]])
    return np.array(out, float)


def box_mesh(boxes):
    """boxes: [(name, center, size), ...] -> verts, faces, group_ranges"""
    V, F, ranges = [], [], []
    off = 0
    for name, center, size in boxes:
        v = box_verts(center, size)
        tris = []
        for q in BOX_QUADS:
            tris.append([q[0], q[1], q[2]])
            tris.append([q[0], q[2], q[3]])
        f = orient_outward(v, np.array(tris, int))   # 逐盒定向, 不能用全局质心
        V.append(v)
        F.append(f + off)
        ranges.append((name, off, off + len(v), len(F) - 1 - len(tris), len(F)))
        off += len(v)
    return np.vstack(V), np.vstack(F), ranges, boxes


def orient_outward(verts, faces):
    """把三角面法向统一朝外（输入须为凸体的三角面）。"""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(1)
    center = verts.mean(0)                        # 凸体的顶点均值 = 内部点
    flip = (n * (cen - center)).sum(1) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def write_obj(path, verts, faces, groups=None, header=""):
    lines = ["# %s" % header] if header else []
    for v in verts:
        lines.append("v %.6f %.6f %.6f" % (v[0], v[1], v[2]))
    gmap = {}
    if groups:
        for name, v0, v1, f0, f1 in groups:
            gmap[name] = (f0, f1)
    if gmap:
        for name, (f0, f1) in gmap.items():
            lines.append("o %s" % name)
            for f in faces[f0:f1]:
                lines.append("f %d %d %d" % (f[0] + 1, f[1] + 1, f[2] + 1))
    else:
        for f in faces:
            lines.append("f %d %d %d" % (f[0] + 1, f[1] + 1, f[2] + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def merge_meshes(parts):
    """parts: [(verts, faces)] -> verts, faces"""
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# 形状池
# --------------------------------------------------------------------------------------

def fib_dirs(n):
    """Fibonacci 球面均匀方向（确定性）。"""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def build_unit_shape(rng, n_dirs, noise_amp, n_waves):
    """
    单位半径「块状球」: 方向采样 -> 径向凸起噪声 -> 凸包。
    凸包保证 | 水密 | 凸 | 顶点数 <= n_dirs。
    """
    dirs = fib_dirs(n_dirs)
    freqs = rng.normal(size=(n_waves, 3)) * 2.2
    phases = rng.uniform(0.0, 2.0 * np.pi, n_waves)
    amps = rng.uniform(0.4, 1.0, n_waves)
    wob = np.sin(dirs @ freqs.T + phases) @ (amps / amps.sum())      # ~[-1, 1]
    r = np.clip(1.0 + noise_amp * wob, 0.6, 1.6)
    pts = dirs * r[:, None]

    hull = ConvexHull(pts)
    idx = hull.vertices
    remap = -np.ones(len(pts), int)
    remap[idx] = np.arange(len(idx))
    verts = pts[idx]
    faces = remap[hull.simplices]
    faces = faces[(faces >= 0).all(1)]
    faces = orient_outward(verts, faces)

    # 用凸包自身重新算体积（= 单位球体积 * 形状系数）
    vol = float(hull.volume)
    return verts, faces, vol


# --------------------------------------------------------------------------------------
# 铺装
# --------------------------------------------------------------------------------------

def sample_sizes(rng, n, cfg):
    d = rng.uniform(*cfg["size_range"], n)
    a = d / 2.0
    b = a * rng.uniform(*cfg["axis_ratio_b"], n)
    c = b * rng.uniform(*cfg["axis_ratio_c"], n)
    return a, b, c


def sample_yaw(rng, n):
    return rng.uniform(0.0, 2.0 * np.pi, n)


def sample_rotations(rng, yaw, tilt_deg):
    """yaw 已定（用于铺装），这里只补倾斜：R = Rz(yaw) * Rtilt。"""
    n = len(yaw)
    tilt = np.radians(tilt_deg) * np.sqrt(rng.uniform(0.0, 1.0, n))
    tdir = rng.uniform(0.0, np.pi, n)          # 倾斜轴方位
    axis = np.stack([np.cos(tdir), np.sin(tdir), np.zeros(n)], axis=1)
    R_tilt = Rotation.from_rotvec(axis * tilt[:, None])
    R_yaw = Rotation.from_rotvec(np.stack([np.zeros(n), np.zeros(n), yaw], axis=1))
    return R_yaw * R_tilt


def ellipse_support(a, b, yaw, phi):
    """椭圆在方向 phi 上的支撑半径（半长轴 a 沿 yaw 方向）。"""
    c = np.cos(phi - yaw)
    s = np.sin(phi - yaw)
    return np.sqrt((a * c) ** 2 + (b * s) ** 2)


def pack_xy(rng, lx, ly, a, b, yaw, coverage, overlap_allow, relax_iters):
    """
    抖动网格初值 + 椭圆支撑函数松弛。
    分离判据: |dx| >= overlap_allow * (h_i(phi) + h_j(phi))，phi 为连心线方向。
    """
    if lx <= 0 or ly <= 0:
        raise ValueError("field size must be positive")
    mean_area = np.pi * float(np.mean(a * b))
    cell = math.sqrt(mean_area / max(coverage, 1e-6))
    nx = max(1, int(np.ceil(lx / cell)))
    ny = max(1, int(np.ceil(ly / cell)))
    gx = -lx / 2 + (np.arange(nx) + 0.5) * (lx / nx)
    gy = -ly / 2 + (np.arange(ny) + 0.5) * (ly / ny)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    n_grid = len(xy)

    # 网格点数与目标石头数不一定相等：多了就随机取子集，少了就补均匀点
    lo = np.array([-lx / 2, -ly / 2])
    hi = np.array([lx / 2, ly / 2])
    n_want = len(a)
    if n_grid > n_want:
        xy = xy[rng.choice(n_grid, size=n_want, replace=False)]
    elif n_grid < n_want:
        extra = rng.uniform(lo, hi, (n_want - n_grid, 2))
        xy = np.vstack([xy, extra])

    j = rng.uniform(-0.5, 0.5, xy.shape) * np.array([lx / nx, ly / ny]) * 0.9
    xy = xy + j

    # 边界钳制（用椭圆在 X/Y 方向上的支撑半径）
    ex = np.sqrt((a * np.cos(yaw)) ** 2 + (b * np.sin(yaw)) ** 2)
    ey = np.sqrt((a * np.sin(yaw)) ** 2 + (b * np.cos(yaw)) ** 2)
    lo_c = np.minimum(lo + np.stack([ex, ey], 1), 0.0)
    hi_c = np.maximum(hi - np.stack([ex, ey], 1), 0.0)
    xy = np.clip(xy, lo_c, hi_c)

    max_r = 2.0 * float(max(a.max(), b.max())) * overlap_allow * 1.01
    for _ in range(relax_iters):
        tree = cKDTree(xy, balanced_tree=False)
        pairs = tree.query_pairs(max_r, output_type="ndarray")
        if len(pairs) == 0:
            break
        i, jj = pairs[:, 0], pairs[:, 1]
        d = xy[jj] - xy[i]
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        phi = np.arctan2(d[:, 1], d[:, 0])
        target = overlap_allow * (ellipse_support(a[i], b[i], yaw[i], phi) +
                                  ellipse_support(a[jj], b[jj], yaw[jj], phi))
        viol = target - dist
        m = viol > 3e-4
        if not m.any():
            break
        push = (d[m] / dist[m][:, None]) * (viol[m] * 0.5 * 0.7)[:, None]
        np.add.at(xy, i[m], -push)
        np.add.at(xy, jj[m], push)
        xy = np.clip(xy, lo_c, hi_c)

    return xy, dict(cell=cell, nx=nx, ny=ny, n_grid=n_grid)


def overlap_report(xy, a, b, yaw, overlap_allow):
    max_r = 2.0 * float(max(a.max(), b.max())) * overlap_allow * 1.01
    tree = cKDTree(xy)
    pairs = tree.query_pairs(max_r, output_type="ndarray")
    if len(pairs) == 0:
        return dict(n_pairs=0, max_penetration=0.0, n_bad=0, mean_penetration=0.0)
    i, j = pairs[:, 0], pairs[:, 1]
    d = xy[j] - xy[i]
    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    phi = np.arctan2(d[:, 1], d[:, 0])
    target = overlap_allow * (ellipse_support(a[i], b[i], yaw[i], phi) +
                              ellipse_support(a[j], b[j], yaw[j], phi))
    pen = np.maximum(target - dist, 0.0)
    return dict(n_pairs=int(len(pairs)),
                max_penetration=float(pen.max()),
                mean_penetration=float(pen.mean()),
                n_bad=int((pen > 0.002).sum()))


# --------------------------------------------------------------------------------------
# 场景几何（槽 / 墙 / 平台 / 站立台 / 填充体）
# --------------------------------------------------------------------------------------

# 只有「站立台尺寸」和「平台宽度」与机器人尺寸相关（要能站下 + 能转身）。
# 卵石层、槽体、台阶高度对所有机型通用 —— 一套场景可给不同机器人共用。
# pad = [沿 X(横向), 沿 Y(前后)] 米; platform_w = 平台宽度(米)
ROBOT_PRESETS = {
    "unitree_go2":   dict(pad=[0.90, 0.60], platform_w=0.85),
    "unitree_a1":    dict(pad=[0.80, 0.55], platform_w=0.80),
    "unitree_b1":    dict(pad=[0.90, 0.60], platform_w=0.90),
    "anymal_c":      dict(pad=[1.10, 0.70], platform_w=1.00),
    "unitree_g1":    dict(pad=[0.90, 0.70], platform_w=0.95),   # 人形: 需转身空间
    "generic_large": dict(pad=[1.20, 1.00], platform_w=1.20),   # 一套通吃(默认)
}


def derive_geometry(cfg):
    t = float(cfg["trough"]["wall_thickness"])
    lx, ly = [float(v) for v in cfg["trough"]["inner_size"]]
    depth = float(cfg["trough"]["depth"])

    plat_w = float(cfg["platform"]["width"])
    plat_top = float(cfg["platform"]["top_height"])

    out = dict(lx=lx, ly=ly, depth=depth, t=t,
               rim_z=depth, floor_z=0.0,
               plat_w=plat_w, plat_top=plat_top,
               plat_x0=lx / 2 + t,                    # 0.80
               plat_x1=lx / 2 + t + plat_w)           # 1.80

    pad = cfg.get("standing_pad") or {}
    pad_size = [float(v) for v in pad.get("size", [0.8, 0.8])]
    pad_th = float(pad.get("thickness", 0.005))

    # 可选: 按机型覆盖"站立台尺寸 + 平台宽度"(只有这两处与机器人尺寸相关;
    # 卵石层/槽体/台阶高度与机型无关, 任何机器人共用一个场景)
    rc = cfg.get("robot_class")
    if rc and rc in ROBOT_PRESETS:
        pr = ROBOT_PRESETS[rc]
        pad_size = [float(v) for v in pr["pad"]]
        plat_w = float(pr["platform_w"])
        out["plat_w"] = plat_w
        out["plat_x1"] = out["plat_x0"] + plat_w
        out["robot_class"] = rc

    pad_c = pad.get("center")
    if pad_c is None:
        cx = (out["plat_x0"] + out["plat_x1"]) / 2
        cy = -ly / 2 + pad_size[1] / 2 + 0.05
        pad_c = [cx, cy]
    out["pad_size"] = pad_size
    out["pad_th"] = pad_th
    out["pad_center"] = [float(pad_c[0]), float(pad_c[1]), plat_top + pad_th / 2]
    return out


def build_boxes(cfg, geo, fill_z, nominal_top):
    lx, ly, t, depth = geo["lx"], geo["ly"], geo["t"], geo["depth"]
    ox, oy = lx + 2 * t, ly + 2 * t
    px0, px1, pt = geo["plat_x0"], geo["plat_x1"], geo["plat_top"]
    pad_c, pad_s = geo["pad_center"], geo["pad_size"]

    boxes = [
        ("trough_floor",  [0.0, 0.0, geo["floor_z"] - 0.025], [ox, oy, 0.05]),
        ("wall_x_minus",  [-(lx / 2 + t / 2), 0.0, depth / 2], [t, oy, depth]),
        ("wall_x_plus",   [+(lx / 2 + t / 2), 0.0, depth / 2], [t, oy, depth]),
        ("wall_y_minus",  [0.0, -(ly / 2 + t / 2), depth / 2], [lx, t, depth]),
        ("wall_y_plus",   [0.0, +(ly / 2 + t / 2), depth / 2], [lx, t, depth]),
        ("platform",      [(px0 + px1) / 2, 0.0, pt / 2], [px1 - px0, oy, pt]),
        ("standing_pad",  [pad_c[0], pad_c[1], pad_c[2]], [pad_s[0], pad_s[1], geo["pad_th"]]),
    ]
    if cfg["pebble_bed"].get("bulk_fill", True):
        boxes.append(("bulk_fill", [0.0, 0.0, fill_z / 2], [lx, ly, fill_z]))
    return boxes


# --------------------------------------------------------------------------------------
# 高度场（椭圆体精确解 z）
# --------------------------------------------------------------------------------------

def rasterize_heightfield(geo, cell, fill_z, pos, rot, semi):
    lx, ly = geo["lx"], geo["ly"]
    nx = int(round(lx / cell))
    ny = int(round(ly / cell))
    origin = np.array([-lx / 2, -ly / 2])
    xs = origin[0] + (np.arange(nx) + 0.5) * cell
    ys = origin[1] + (np.arange(ny) + 0.5) * cell
    H = np.full((nx, ny), fill_z, np.float32)

    for k in range(len(pos)):
        R = rot[k].as_matrix()
        M = R @ np.diag(1.0 / np.asarray(semi[k], float) ** 2) @ R.T
        cx, cy, cz = pos[k]
        ex = float(np.sqrt(max(M[0, 0], 1e-12)))
        # 保守包围: 用最大半轴
        rad = float(max(semi[k])) * 1.02
        i0 = max(0, int(np.searchsorted(xs, cx - rad)))
        i1 = min(nx, int(np.searchsorted(xs, cx + rad)))
        j0 = max(0, int(np.searchsorted(ys, cy - rad)))
        j1 = min(ny, int(np.searchsorted(ys, cy + rad)))
        if i1 <= i0 or j1 <= j0:
            continue
        dx = xs[i0:i1][:, None] - cx
        dy = ys[j0:j1][None, :] - cy
        A = M[2, 2]
        B = 2.0 * (M[2, 0] * dx + M[2, 1] * dy)
        C = M[0, 0] * dx ** 2 + 2.0 * M[0, 1] * dx * dy + M[1, 1] * dy ** 2 - 1.0
        disc = B * B - 4.0 * A * C
        ok = disc > 0.0
        if not ok.any():
            continue
        u = (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
        z = np.where(ok, cz + u, -np.inf)
        sub = H[i0:i1, j0:j1]
        np.maximum(sub, z, out=sub)
        H[i0:i1, j0:j1] = sub
    return H


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def default_config():
    return {
        "units": "m", "up_axis": "Z", "quat_order": "wxyz",
        "seed": 20261019,
        "robot_class": "generic_large",
        "trough": {"inner_size": [1.5, 4.0], "depth": 0.20, "wall_thickness": 0.05},
        "platform": {"width": 1.20, "top_height": 0.25, "side": "+x"},
        "standing_pad": {"size": [1.20, 1.00], "center": None, "thickness": 0.005},
        "pebble_bed": {"nominal_top": 0.15, "bulk_fill": True},
        "pebbles": {
            "size_range": [0.020, 0.060],
            "axis_ratio_b": [0.60, 0.90],
            "axis_ratio_c": [0.50, 0.80],
            "coverage": 0.75,
            "tilt_deg": 20.0,
            "noise_amp": 0.07,
            "n_waves": 6,
            "pool_size": 32,
            "n_dirs": 32,
            "overlap_allow": 0.95,
            "relax_iters": 150,
            "z_jitter": 0.0015,
        },
        "materials": {
            "density": 2600.0,
            "mu_mean": 0.55,
            "mu_range": [0.40, 0.70],
        },
        "output": {
            "n_layouts": 32,
            "heightfield_cell": 0.005,
            "static_mesh_layouts": [0],
            "preview": True,
        },
    }


def deep_update(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def generate_layout(rng, cfg, geo, pool_vol, shapes_nf, nominal_top):
    pc = cfg["pebbles"]
    lx, ly = geo["lx"], geo["ly"]

    # 先估点数：按目标覆盖率和平均尺寸
    a_tmp, b_tmp, _ = sample_sizes(rng, 4096, pc)
    mean_area = np.pi * float(np.mean(a_tmp * b_tmp))
    n_est = int(round(lx * ly * pc["coverage"] / mean_area))
    n_est = max(1, n_est)

    a, b, c = sample_sizes(rng, n_est, pc)
    yaw = sample_yaw(rng, n_est)
    xy, grid_info = pack_xy(rng, lx, ly, a, b, yaw, pc["coverage"],
                            pc["overlap_allow"], pc["relax_iters"])
    n = len(xy)
    a, b, c, yaw = a[:n], b[:n], c[:n], yaw[:n]

    rot = sample_rotations(rng, yaw, pc["tilt_deg"])
    # 竖直半高（椭球精确值）
    Rm = rot.as_matrix()                       # (n,3,3)
    semi = np.stack([a, b, c], axis=1)         # (n,3)
    z_ext = np.linalg.norm(Rm[:, 2, :] * semi, axis=1)

    fill_z = float(nominal_top - 2.0 * z_ext.max())
    z = fill_z + z_ext + rng.uniform(-pc["z_jitter"], pc["z_jitter"], n)
    # 不允许沉进填充体
    z = np.maximum(z, fill_z + z_ext * 0.9)

    scale = semi
    shape_id = rng.integers(0, len(pool_vol), n).astype(np.int16)
    vol = pool_vol[shape_id] * (a * b * c)
    mass = vol * cfg["materials"]["density"]
    mu = rng.uniform(*cfg["materials"]["mu_range"], n)

    q = rot.as_quat()                          # scipy: (x, y, z, w)
    quat_wxyz = np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], axis=1)

    pos = np.stack([xy[:, 0], xy[:, 1], z], axis=1)
    rep = overlap_report(xy, a, b, yaw, pc["overlap_allow"])
    cover = float(np.sum(np.pi * a * b) / (lx * ly))
    info = dict(
        n=int(n), n_est=int(n_est), fill_z=fill_z,
        top_min=float((z + z_ext).min()), top_max=float((z + z_ext).max()),
        top_mean=float((z + z_ext).mean()),
        mass_min=float(mass.min()), mass_max=float(mass.max()), mass_sum=float(mass.sum()),
        mass_mean=float(mass.mean()), coverage=cover, grid=grid_info, overlap=rep,
    )
    return dict(pos=pos, quat=quat_wxyz, scale=scale, shape_id=shape_id,
                mass=mass, mu=mu, a=a, b=b, c=c, rot=rot, semi=semi, info=info)


def make_preview(path, cfg, geo, layout, H, hf_meta, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  [warn] preview skipped:", e)
        return

    lx, ly = geo["lx"], geo["ly"]
    px0, px1, pt = geo["plat_x0"], geo["plat_x1"], geo["plat_top"]
    pad_c, pad_s = geo["pad_center"], geo["pad_size"]
    t = geo["t"]
    pos = layout["pos"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.add_patch(plt.Rectangle((-lx / 2 - t, -ly / 2 - t), lx + 2 * t, ly + 2 * t,
                               fill=False, lw=2, ec="k"))
    ax.add_patch(plt.Rectangle((-lx / 2, -ly / 2), lx, ly, fill=False, lw=1, ec="0.4"))
    ax.add_patch(plt.Rectangle((px0, -ly / 2 - t), px1 - px0, ly + 2 * t,
                               fc="0.85", ec="k", lw=1.5, label="platform (%.3f m)" % pt))
    ax.add_patch(plt.Rectangle((pad_c[0] - pad_s[0] / 2, pad_c[1] - pad_s[1] / 2),
                               pad_s[0], pad_s[1], fc="tab:orange", ec="tab:red", lw=1.2,
                               alpha=0.8, label="standing pad"))
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=(layout["a"] * 2000), c=pos[:, 2],
                    cmap="YlOrBr", ec="none")
    plt.colorbar(sc, ax=ax, label="pebble center z [m]")
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("top view: %d pebbles, coverage %.2f" % (layout["info"]["n"],
                                                          layout["info"]["coverage"]))
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    if H is not None:
        j = H.shape[1] // 2
        xs = hf_meta["origin"][0] + (np.arange(H.shape[0]) + 0.5) * hf_meta["cell"]
        ax2.fill_between(xs, 0, H[:, j], color="0.75", step="mid", label="heightfield (y=0)")
    ax2.axhline(layout["info"]["fill_z"], color="tab:brown", ls="--", lw=1, label="bulk fill")
    ax2.plot([-lx / 2 - t, -lx / 2], [geo["depth"], geo["depth"]], "k", lw=3)
    ax2.plot([px0, px1], [pt, pt], "k", lw=4)
    ax2.plot([-lx / 2, -lx / 2], [0, geo["depth"]], "k", lw=2)
    ax2.plot([px0, px0], [0, pt], "k", lw=2)
    ax2.set_xlim(-lx / 2 - 0.15, px1 + 0.15)
    ax2.set_ylim(0, max(pt, geo["depth"]) + 0.08)
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Z [m]")
    ax2.set_title("cross-section at Y=0")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="generate cobblestone trough + platform scene bundle")
    ap.add_argument("--config", default=None, help="JSON 配置（缺省用内置默认值）")
    ap.add_argument("--out", default="out_bundle", help="输出目录")
    ap.add_argument("--layouts", type=int, default=None, help="覆盖 output.n_layouts")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-static-mesh", action="store_true")
    args = ap.parse_args(argv)

    cfg = default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8-sig") as fh:   # 兼容 Windows 的 BOM
            deep_update(cfg, json.load(fh))
    if args.layouts is not None:
        cfg["output"]["n_layouts"] = args.layouts

    t0 = time.time()
    out = args.out
    os.makedirs(out, exist_ok=True)
    for sub in ("shapes", "static"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    geo = derive_geometry(cfg)
    nominal_top = float(cfg["pebble_bed"]["nominal_top"])
    lx, ly = geo["lx"], geo["ly"]
    print("[scene] trough %.2f x %.2f m, depth %.3f, platform top %.3f (x %.2f..%.2f)"
          % (lx, ly, geo["depth"], geo["plat_top"], geo["plat_x0"], geo["plat_x1"]))

    # ---- 形状池 ----------------------------------------------------------------
    rng_pool = np.random.default_rng(int(cfg["seed"]))
    pc = cfg["pebbles"]
    pool = []
    for k in range(int(pc["pool_size"])):
        v, f, vol = build_unit_shape(rng_pool, int(pc["n_dirs"]),
                                     float(pc["noise_amp"]), int(pc["n_waves"]))
        pool.append((v, f, vol))
    sizes = np.array([len(p[0]) for p in pool])
    verts_all, faces_all, offsets = [], [], [0]
    for v, f, _ in pool:
        verts_all.append(v)
        faces_all.append(f + offsets[-1])
        offsets.append(offsets[-1] + len(v))
    verts_all = np.vstack(verts_all).astype(np.float32)
    faces_all = np.vstack(faces_all).astype(np.int32)
    pool_vol = np.array([p[2] for p in pool])
    np.savez_compressed(os.path.join(out, "shapes", "rock_pool.npz"),
                        verts=verts_all, faces=faces_all,
                        offsets=np.array(offsets, np.int32),
                        volume=pool_vol.astype(np.float32))
    write_obj(os.path.join(out, "shapes", "rock_pool.obj"), verts_all, faces_all,
              header="unit convex shapes (scale per instance)")
    print("[shapes] pool=%d, verts/shape=%d..%d, unit volume %.3f..%.3f"
          % (len(pool), sizes.min(), sizes.max(), pool_vol.min(), pool_vol.max()))

    # ---- 逐布局 ----------------------------------------------------------------
    rng_layout = np.random.default_rng(int(cfg["seed"]) + 12345)
    n_layouts = int(cfg["output"]["n_layouts"])
    layouts = []
    stat_boxes = None
    hf_meta = None
    for li in range(n_layouts):
        lay = generate_layout(rng_layout, cfg, geo, pool_vol, None, nominal_top)
        # 用本布局的真实 z_ext 推导 fill_z（保证最大卵石顶端 = nominal_top）
        layouts.append(lay)
        i = lay["info"]
        print("  [layout %02d] n=%4d top=[%.3f,%.3f] mean=%.3f cover=%.2f "
              "mass=%.1f..%.1f g overlap_max=%.1f mm bad=%d"
              % (li, i["n"], i["top_min"], i["top_max"], i["top_mean"], i["coverage"],
                 1000 * i["mass_min"], 1000 * i["mass_max"],
                 1000 * i["overlap"]["max_penetration"], i["overlap"]["n_bad"]))

        if stat_boxes is None:
            stat_boxes = build_boxes(cfg, geo, i["fill_z"], nominal_top)

        # 高度场
        H = rasterize_heightfield(geo, float(cfg["output"]["heightfield_cell"]),
                                  i["fill_z"], lay["pos"], lay["rot"], lay["semi"])
        np.save(os.path.join(out, "static", "heightfield_L%02d.npy" % li), H.astype(np.float32))
        if hf_meta is None:
            hf_meta = dict(origin=[-lx / 2, -ly / 2],
                           cell=float(cfg["output"]["heightfield_cell"]),
                           shape=[int(H.shape[0]), int(H.shape[1])],
                           order="[i, j]: i->X, j->Y, origin at (-lx/2, -ly/2), cell centers",
                           model="noise-free ellipsoid upper envelope (approx); merged mesh is authoritative")

        # 合并卵石网格（仅指定布局）
        if (li in cfg["output"].get("static_mesh_layouts", []) and not args.no_static_mesh):
            Rm = lay["rot"].as_matrix()
            parts = []
            base_v, base_f, base_off = pool[0][0], pool[0][1], None
            for k in range(lay["info"]["n"]):
                sid = int(lay["shape_id"][k])
                v = pool[sid][0] * lay["scale"][k]
                v = v @ Rm[k].T + lay["pos"][k]
                parts.append((v, pool[sid][1]))
            mv, mf = merge_meshes(parts)
            write_obj(os.path.join(out, "static", "cobble_L%02d.obj" % li), mv, mf,
                      header="merged cobble surface, layout %02d (single layer)" % li)

        if cfg["output"].get("preview", True) and not args.no_preview and li == 0:
            make_preview(os.path.join(out, "preview.png"), cfg, geo, lay, H, hf_meta,
                         "cobblestone trough %.1f x %.1f m" % (lx, ly))

    # ---- 静态几何 ---------------------------------------------------------------
    V, F, groups, _ = box_mesh([(n, c, s) for n, c, s in stat_boxes])
    write_obj(os.path.join(out, "static", "stage.obj"), V, F, groups=groups,
              header="trough walls / platform / standing pad / bulk fill")
    with open(os.path.join(out, "static", "stage_boxes.json"), "w", encoding="utf-8") as fh:
        json.dump([{"name": n, "center": [float(x) for x in c],
                    "size": [float(x) for x in s]} for n, c, s in stat_boxes],
                  fh, indent=2)
    with open(os.path.join(out, "static", "heightfield_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(hf_meta, fh, indent=2)

    # ---- 打包 layouts -----------------------------------------------------------
    npz = dict(
        n_layouts=np.int32(n_layouts),
        n_per_layout=np.array([l["info"]["n"] for l in layouts], np.int32),
        fill_z=np.array([l["info"]["fill_z"] for l in layouts], np.float32),
        shape_id=np.concatenate([l["shape_id"] for l in layouts]),
        pos=np.concatenate([l["pos"] for l in layouts]).astype(np.float32),
        quat=np.concatenate([l["quat"] for l in layouts]).astype(np.float32),
        scale=np.concatenate([l["scale"] for l in layouts]).astype(np.float32),
        mass=np.concatenate([l["mass"] for l in layouts]).astype(np.float32),
        mu=np.concatenate([l["mu"] for l in layouts]).astype(np.float32),
    )
    np.savez_compressed(os.path.join(out, "layouts.npz"), **npz)

    # ---- manifest ---------------------------------------------------------------
    all_info = [l["info"] for l in layouts]
    manifest = {
        "bundle_version": "1.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": {"script": "gen_pebbles.py", "seed": int(cfg["seed"])},
        "units": cfg["units"], "up_axis": cfg["up_axis"], "quat_order": "wxyz",
        "coordinate_frame": {
            "trough_inner_x": [-lx / 2, lx / 2],
            "trough_inner_y": [-ly / 2, ly / 2],
            "floor_z": geo["floor_z"], "rim_z": geo["rim_z"],
            "platform_x": [geo["plat_x0"], geo["plat_x1"]],
            "platform_top_z": geo["plat_top"],
            "pebble_nominal_top_z": nominal_top,
            "bulk_fill_top_z": float(np.mean([i["fill_z"] for i in all_info])),
            "standing_pad_center": geo["pad_center"],
            "standing_pad_size": geo["pad_size"],
            "suggested_spawn": {
                "pos": [geo["pad_center"][0], geo["pad_center"][1], geo["plat_top"] + geo["pad_th"]],
                "yaw_deg": 90.0,
                "note": "机器人站在平台上, 需向 -X 跨下约 0.11 m 进入卵石区, 然后沿 +Y 前进",
            },
        },
        "shapes": {"count": int(pc["pool_size"]), "n_dirs": int(pc["n_dirs"]),
                   "verts_range": [int(sizes.min()), int(sizes.max())],
                   "file": "shapes/rock_pool.npz",
                   "sha256": sha256_file(os.path.join(out, "shapes", "rock_pool.npz"))},
        "layouts": {"n": n_layouts,
                    "n_per_layout": [int(i["n"]) for i in all_info],
                    "file": "layouts.npz",
                    "sha256": sha256_file(os.path.join(out, "layouts.npz")),
                    "total_pebbles": int(sum(i["n"] for i in all_info))},
        "physics_hints": {
            "dt_recommended": 0.001,
            "position_iterations": 8,
            "contact_offset": 0.002,
            "rest_offset": 0.001,
            "mu_pebble_pebble": [float(v) for v in cfg["materials"]["mu_range"]],
            "mu_pebble_rubber": [0.80, 1.00],
            "mu_pebble_mortar": 0.45,
            "restitution": 0.10,
            "density": float(cfg["materials"]["density"]),
            "mass_mean_g": float(np.mean([i["mass_mean"] for i in all_info]) * 1000),
            "note": "卵石为单层、半埋于 bulk_fill 面; 动力学版请只给少 env 实例化",
        },
        "quality": {
            "coverage_mean": float(np.mean([i["coverage"] for i in all_info])),
            "overlap_max_mm": float(max(i["overlap"]["max_penetration"] for i in all_info) * 1000),
            "overlap_bad_total": int(sum(i["overlap"]["n_bad"] for i in all_info)),
            "top_mean_range": [float(min(i["top_mean"] for i in all_info)),
                               float(max(i["top_mean"] for i in all_info))],
        },
        "files": {
            "stage_boxes": "static/stage_boxes.json",
            "stage_mesh": "static/stage.obj",
            "cobble_mesh_layouts": cfg["output"].get("static_mesh_layouts", []),
            "heightfield": "static/heightfield_L%02d.npy",
            "heightfield_meta": "static/heightfield_meta.json",
        },
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=True)
    eff = json.loads(json.dumps(cfg))
    eff["_derived"] = geo
    with open(os.path.join(out, "layout_config.json"), "w", encoding="utf-8") as fh:
        json.dump(eff, fh, indent=2, ensure_ascii=True)

    dt = time.time() - t0
    sz = sum(os.path.getsize(os.path.join(r, f))
             for r, _, fs in os.walk(out) for f in fs)
    print("[done] %d layouts, %d pebbles total, %.1f MB, %.1f s -> %s"
          % (n_layouts, manifest["layouts"]["total_pebbles"], sz / 1e6, dt, os.path.abspath(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
