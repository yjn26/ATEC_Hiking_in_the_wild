# -*- coding: utf-8 -*-
"""临时校验: 用精确椭圆画出局部窗口, 检查铺装是否成团/有空隙。"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree

bundle = sys.argv[1] if len(sys.argv) > 1 else "pebble_scene/_preview"
li = int(sys.argv[2]) if len(sys.argv) > 2 else 0
z = np.load(bundle + "/layouts.npz")
npl = z["n_per_layout"]
off = np.concatenate([[0], np.cumsum(npl)])
s, e = off[li], off[li + 1]
pos, scale, quat = z["pos"][s:e], z["scale"][s:e], z["quat"][s:e]
print("layout %d / %d   n = %d" % (li, len(npl), len(pos)))

# yaw from quaternion (wxyz) 的 z 旋转分量（倾斜很小, 近似）
w, x, y, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
yaw = np.arctan2(2 * (w * qz + x * y), 1 - 2 * (y * y + qz * qz))

tree = cKDTree(pos[:, :2])
d, _ = tree.query(pos[:, :2], k=2)
nn = d[:, 1]
print("nearest-neighbour dist: min %.4f  mean %.4f  p95 %.4f" % (nn.min(), nn.mean(), np.percentile(nn, 95)))

# 空隙检测: 2cm 网格上未被椭圆覆盖的格子占比
g = 0.02
xe = np.arange(-0.75, 0.75, g)
ye = np.arange(-2.0, 2.0, g)
GX, GY = np.meshgrid(xe, ye, indexing="ij")
cov = np.zeros(GX.shape, bool)
for k in range(len(pos)):
    a, b = scale[k, 0], scale[k, 1]
    ca, sa = np.cos(-yaw[k]), np.sin(-yaw[k])
    dx = GX - pos[k, 0]
    dy = GY - pos[k, 1]
    u = dx * ca - dy * sa
    v = dx * sa + dy * ca
    cov |= (u / a) ** 2 + (v / b) ** 2 <= 1.0
print("grid-void fraction (%.0f mm cells uncovered): %.3f" % (g * 1000, 1 - cov.mean()))

fig, ax = plt.subplots(figsize=(7, 7))
cx, cy = 0.0, 0.0
for k in range(len(pos)):
    if abs(pos[k, 0] - cx) < 0.3 and abs(pos[k, 1] - cy) < 0.3:
        e = Ellipse((pos[k, 0], pos[k, 1]), 2 * scale[k, 0], 2 * scale[k, 1],
                    angle=np.degrees(yaw[k]) + 90.0,  # 半轴 a 沿 yaw, 而 Ellipse 的 width 沿 angle
                    fc="tab:brown", ec="k", lw=0.4, alpha=0.85)
        ax.add_patch(e)
ax.set_xlim(cx - 0.3, cx + 0.3)
ax.set_ylim(cy - 0.3, cy + 0.3)
ax.set_aspect("equal")
ax.set_title("exact ellipses, 0.6 x 0.6 m window")
fig.tight_layout()
out_png = bundle + "/pack_check.png"
fig.savefig(out_png, dpi=140)
print("saved", out_png)
