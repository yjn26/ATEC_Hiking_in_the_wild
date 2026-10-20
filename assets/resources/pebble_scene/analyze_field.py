# -*- coding: utf-8 -*-
"""analyze_field.py —— 用「露出硬底板（bulk_fill）的面积比」判断机器人有没有把卵石挤开、
最后踩在槽底硬面上。

做法：把卵石顶面包络栅格化成 1 cm 高度场（复用 gen_pebbles.rasterize_heightfield），
某格子 H < fill_z + 6 mm 即视为「硬底板露出来」。分别算：
  * 全槽
  * 机器人走过的走廊（到 traj 任一点 XY 距离 <= 0.35 m 的格子）

用法:
  python analyze_field.py <bundle_dir> <layouts.npz> [field_final.npz]
"""
import io
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pebble_scene"))
import gen_pebbles as gp  # noqa: E402

LX, LY = 1.5, 4.0
GEO = dict(lx=LX, ly=LY, depth=0.20)
CELL = 0.01


def load_layouts(path, li=0):
    z = np.load(path)
    off = np.concatenate([[0], np.cumsum(z["n_per_layout"])])
    s, e = int(off[li]), int(off[li + 1])
    return dict(pos=z["pos"][s:e].astype(np.float64), quat=z["quat"][s:e].astype(np.float64),
                scale=z["scale"][s:e].astype(np.float64), n=e - s)


def heightmap(pos, quat, scale, fill_z):
    rot = Rotation.from_quat(np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=1))
    return gp.rasterize_heightfield(GEO, CELL, fill_z, pos, rot, scale)


def corridor_mask(traj, radius=0.35):
    nx, ny = int(round(LX / CELL)), int(round(LY / CELL))
    xs = -LX / 2 + (np.arange(nx) + 0.5) * CELL
    ys = -LY / 2 + (np.arange(ny) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    m = np.zeros((nx, ny), bool)
    if traj is None or len(traj) == 0:
        return m
    t = np.asarray(traj, float).reshape(-1, 2)
    # 细分成 2 cm 的折线采样，避免点稀疏时走廊断成点
    pts = [t[0]]
    for a, b in zip(t[:-1], t[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.02)))
        for i in range(1, k + 1):
            pts.append(a + (b - a) * i / k)
    for p in np.asarray(pts):
        m |= ((X - p[0]) ** 2 + (Y - p[1]) ** 2) <= radius ** 2
    return m


def report(H, fill_z, mask, label, expose_mm=6.0):
    from scipy.ndimage import binary_opening
    ex = H <= (fill_z + expose_mm / 1000.0)
    # 开运算：细缝（2~4 mm）不算“露底”，只有半径 ≥1 cm 的洞才算——那才是脚踩得到/眼看得见的池底
    r = max(1, int(round(0.010 / CELL)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    wide = binary_opening(ex, structure=(xx ** 2 + yy ** 2) <= r ** 2)
    tot = ex.size if mask is None or not mask.any() else int(mask.sum())
    n_ex = int(ex.sum()) if mask is None or not mask.any() else int((ex & mask).sum())
    n_wd = int(wide.sum()) if mask is None or not mask.any() else int((wide & mask).sum())
    sub = H if mask is None or not mask.any() else H[mask]
    print("  %-22s exposed=%6.3f%%  wide=%6.3f%%  H: mean=%.1f mm  p5=%.1f  p50=%.1f"
          % (label, 100.0 * n_ex / max(tot, 1), 100.0 * n_wd / max(tot, 1),
             1000 * (sub.mean() - fill_z), 1000 * (np.percentile(sub, 5) - fill_z),
             1000 * (np.percentile(sub, 50) - fill_z)))
    return 100.0 * n_ex / max(tot, 1), 100.0 * n_wd / max(tot, 1)


def main():
    bundle, lay_path = sys.argv[1], sys.argv[2]
    field = sys.argv[3] if len(sys.argv) > 3 else None
    man = json.load(io.open(os.path.join(bundle, "manifest.json"), encoding="utf-8"))
    fill_z = float(man["coordinate_frame"]["bulk_fill_top_z"])
    L = load_layouts(lay_path)
    print("bundle=%s n=%d fill_z=%.4f" % (bundle, L["n"], fill_z))

    H0 = heightmap(L["pos"], L["quat"], L["scale"], fill_z)
    print("[before] (as authored)")
    report(H0, fill_z, None, "whole trough")

    if not field:
        return
    d = np.load(field)
    pos1 = d["pos"].astype(np.float64)
    traj = d["traj"] if "traj" in d else None
    print("[field] dumped n=%d traj=%d pts fill_z=%.4f" % (len(pos1), 0 if traj is None else len(traj), float(d["fill_z"])))
    if len(pos1) != L["n"]:
        print("  !! count mismatch vs bundle (%d vs %d)" % (len(pos1), L["n"]))
        return
    # 沉降后位置：用 bundle 的旋转/尺度近似（石头基本只平移）
    H1 = heightmap(pos1, L["quat"], L["scale"], float(d["fill_z"]))
    mask = corridor_mask(traj)
    print("[after]  (post-run)")
    report(H1, float(d["fill_z"]), None, "whole trough")
    report(H1, float(d["fill_z"]), mask, "walked corridor")
    report(H0, float(d["fill_z"]), mask, "corridor BEFORE run")
    disp = d["disp_mm"]
    print("  displacement: >2mm=%.1f%%  >10mm=%.1f%%  >50mm=%.1f%%  max=%.0f mm"
          % (100.0 * (disp > 2).mean(), 100.0 * (disp > 10).mean(), 100.0 * (disp > 50).mean(), disp.max()))
    if traj is not None and len(traj):
        print("  robot path: start=(%.2f,%.2f) end=(%.2f,%.2f)  len≈%.2f m"
              % (traj[0][0], traj[0][1], traj[-1][0], traj[-1][1],
                 np.linalg.norm(np.diff(np.asarray(traj, float), axis=0), axis=1).sum()))


if __name__ == "__main__":
    main()
