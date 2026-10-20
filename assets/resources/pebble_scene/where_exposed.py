# -*- coding: utf-8 -*-
"""where_exposed.py —— 露底格子分布在哪儿（贴墙 vs 槽心 vs 脚窝）。"""
import numpy as np
import sys
from scipy.ndimage import binary_opening
from scipy.spatial.transform import Rotation

sys.path.insert(0, "pebble_scene")
import gen_pebbles as gp

LX, LY, CELL = 1.5, 4.0, 0.01
GEO = dict(lx=LX, ly=LY, depth=0.20)


def H_of(bundle, pos, quat, scale, fill):
    rot = Rotation.from_quat(np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=1))
    return gp.rasterize_heightfield(GEO, CELL, fill, pos, rot, scale)


bundle, field = sys.argv[1], sys.argv[2]
z = np.load(bundle + "/layouts.npz")
off = np.concatenate([[0], np.cumsum(z["n_per_layout"])]).astype(int)
s, e = int(off[0]), int(off[1])
quat, scale = z["quat"][s:e], z["scale"][s:e]
d = np.load(field)
pos = d["pos"].astype(np.float64)
fill = float(d["fill_z"])
traj = d["traj"]

H = H_of(bundle, pos, quat, scale, fill)
ex = H <= fill + 0.006
r = 1
yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
wide = binary_opening(ex, structure=(xx ** 2 + yy ** 2) <= r ** 2)

nx, ny = H.shape
xs = -LX / 2 + (np.arange(nx) + 0.5) * CELL
ys = -LY / 2 + (np.arange(ny) + 0.5) * CELL
X, Y = np.meshgrid(xs, ys, indexing="ij")
wall = np.minimum.reduce([X - (-LX / 2), LX / 2 - X, Y - (-LY / 2), LY / 2 - Y])

# 走廊
pts = [traj[0]]
for a, b in zip(traj[:-1], traj[1:]):
    k = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.02)))
    for i in range(1, k + 1):
        pts.append(a + (b - a) * i / k)
pts = np.asarray(pts)
mask = np.zeros_like(wide)
for p in pts:
    mask |= ((X - p[0]) ** 2 + (Y - p[1]) ** 2) <= 0.35 ** 2

print("bundle=%s n=%d fill=%.4f" % (bundle, len(pos), fill))
print("corridor wide holes: %d cells (%.3f%% of corridor)" % ((wide & mask).sum(), 100 * (wide & mask).sum() / mask.sum()))
for lo, hi in ((0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.0)):
    sel = (wall >= lo) & (wall < hi)
    if sel.sum() == 0:
        continue
    print("  wall distance %.2f-%.2f m: wide %.3f%%  (cells %d)"
          % (lo, hi, 100 * (wide & sel).sum() / sel.sum(), sel.sum()))
# 洞的大小分布
from scipy.ndimage import label
lab, n = label(wide & mask)
sizes = np.bincount(lab.ravel())[1:]
if len(sizes):
    print("  hole count=%d, area cells: median=%d, max=%d (=%.1f cm across)"
          % (len(sizes), int(np.median(sizes)), int(sizes.max()), 2 * np.sqrt(sizes.max()) * CELL * 100))
