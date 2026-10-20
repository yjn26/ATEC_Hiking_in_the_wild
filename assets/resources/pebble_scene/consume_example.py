#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consume_example.py
==================
读取 gen_pebbles.py 产生的 bundle，并给出两套引擎的接入示例。
本文件只做「读取 + 参数整理」，真正建仿真部分以注释形式给出，需要在装了
Isaac Lab / mjlab 的机器上补全（这里的环境没有这两个包）。

bundle 约定
-----------
* 坐标系: Z-up, 米, 四元数 (w, x, y, z)
* 池内形状为「单位半径块状凸包」; 每实例的 scale 即三个半轴 (a, b, c) [m]
* 卵石为单层, 半埋于 bulk_fill 顶面; 下方不需要再建石头
"""
from __future__ import annotations

import json
import os
import numpy as np


# --------------------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------------------

class PebbleBundle:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "manifest.json"), "r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        z = np.load(os.path.join(root, "shapes", "rock_pool.npz"))
        self.pool_verts = z["verts"]                       # (V, 3) float32
        self.pool_faces = z["faces"]                       # (F, 3) int32
        self.pool_offsets = z["offsets"]                   # (n_shape+1,)
        self.pool_volume = z["volume"]                     # 单位体积
        z = np.load(os.path.join(root, "layouts.npz"))
        self.n_per_layout = z["n_per_layout"]
        self.layout_offsets = np.concatenate([[0], np.cumsum(self.n_per_layout)])
        self.shape_id = z["shape_id"]
        self.pos = z["pos"]                                # (N, 3)
        self.quat = z["quat"]                              # (N, 4) wxyz
        self.scale = z["scale"]                            # (N, 3) 半轴
        self.mass = z["mass"]
        self.mu = z["mu"]
        self.fill_z = z["fill_z"]
        with open(os.path.join(root, "static", "stage_boxes.json"), "r", encoding="utf-8") as fh:
            self.stage_boxes = json.load(fh)

    # ---- 形状 ------------------------------------------------------------------
    def shape(self, i):
        """第 i 个池形状的 (verts, faces)。"""
        v0, v1 = self.pool_offsets[i], self.pool_offsets[i + 1]
        f0 = self.pool_faces
        mask = (f0 >= v0).all(1) & (f0 < v1).all(1)
        return self.pool_verts[v0:v1], f0[mask] - v0

    # ---- 布局 ------------------------------------------------------------------
    def layout(self, li):
        s, e = self.layout_offsets[li], self.layout_offsets[li + 1]
        return dict(pos=self.pos[s:e], quat=self.quat[s:e], scale=self.scale[s:e],
                    shape_id=self.shape_id[s:e], mass=self.mass[s:e], mu=self.mu[s:e],
                    fill_z=float(self.fill_z[li]))

    def heightfield(self, li):
        p = os.path.join(self.root, "static", "heightfield_L%02d.npy" % li)
        with open(os.path.join(self.root, "static", "heightfield_meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return np.load(p), meta

    # ---- 子集 ------------------------------------------------------------------
    def subset(self, li, x=None, y=None):
        """"只取脚下走廊"之类的空间筛选, 返回索引数组（用于动态石头）。"""
        lay = self.layout(li)
        m = np.ones(len(lay["pos"]), bool)
        if x is not None:
            m &= (lay["pos"][:, 0] >= x[0]) & (lay["pos"][:, 0] <= x[1])
        if y is not None:
            m &= (lay["pos"][:, 1] >= y[0]) & (lay["pos"][:, 1] <= y[1])
        return np.nonzero(m)[0]


# --------------------------------------------------------------------------------------
# Isaac Lab 接入示例（伪代码，需在有 isaaclab 的机器上补全）
# --------------------------------------------------------------------------------------
ISAAC_SNIPPET = r'''
# ---- 1) 静态几何: 直接用 stage_boxes.json 建 CuboidCfg, 不要用网格 ----
import json, isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.utils import configclass

boxes = json.load(open(f"{BUNDLE}/static/stage_boxes.json"))
stage_cfgs = {
    b["name"]: RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Stage/{b['name']}",
        spawn=sim_utils.CuboidCfg(
            size=tuple(b["size"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                         disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.001),
            mass_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(b["center"]), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    for b in boxes
}

# ---- 2) 动态卵石: RigidObjectCollection, 每 env 一份布局, 用下标取布局 ----
# 注意 prim 数 = num_envs * n_pebble, 训练时务必只给少数 env 开动态石头
P = PebbleBundle(BUNDLE)
lay = P.layout(0)
pebble_cfgs = {}
for k in range(len(lay["pos"])):
    v, f = P.shape(int(lay["shape_id"][k]))
    pebble_cfgs[f"rock_{k:04d}"] = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Pebbles/rock_{k:04d}",
        spawn=sim_utils.MeshConverterCfg(...) ,  # 或先落盘 .obj 再 UsdFileCfg(usd_path=...)
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(lay["pos"][k]),
            rot=tuple(lay["quat"][k]),          # wxyz, Isaac 与 MuJoCo 同序
        ),
    )
pebbles = RigidObjectCollectionCfg(rigid_objects=pebble_cfgs)

# ---- 3) 质量/摩擦/阻尼: 显式覆盖, 不要靠引擎按尺寸猜 ----
# pebbles.write_object_state_to_sim(state) 逐 env 复位; 或启动后一次性 set mass
# (Isaac Lab 里改质量走 prim 上的 massAPI; 摩擦走 material 绑定)

# ---- 4) 每 episode reset 卵石（按 env 掩码，只重置该 env 的石头）----
# pebbles.write_object_state_to_sim(default_state, env_ids=env_ids)
'''

# --------------------------------------------------------------------------------------
# mjlab / 纯 MuJoCo 接入示例（伪代码）
# --------------------------------------------------------------------------------------
MJLAB_SNIPPET = r'''
# ---- 1) 静态几何: box 直接写 geom, 比 mesh 便宜得多 ----
#   <geom name="wall_x_minus" type="box" pos="..." size="hx hy hz" rgba=".5 .5 .5"/>
#   注意 MuJoCo 的 box size 是「半长」, stage_boxes.json 里是「全长」, 记得 /2

# ---- 2) 卵石: 每颗一个 freejoint body + 1 个 mesh geom ----
#   spec = mujoco.MjSpec()
#   for k in range(n):
#       b = spec.worldbody.add_body(pos=pos[k], quat=quat[k])   # mjSpec 的 quat 也是 wxyz
#       b.add_freejoint()
#       b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, mesh=mesh_pool[shape_id[k]],
#                  mass=mass[k], friction=(mu[k], 0.005, 0.0001),
#                  solref=(0.004, 1.0), solimp=(0.9, 0.95, 0.001))
#   非均匀缩放: MuJoCo 的 mesh 不支持逐实例 scale, 所以要按 scale 把顶点烘进
#   池内形状（生成时就按 scale 生成一份）, 或者用 sphere 近似。

# ---- 3) 静态地形替代方案（推荐大规模训练用）----
#   h = np.load(f"{BUNDLE}/static/heightfield_L00.npy")   # [nx, ny], i->X, j->Y
#   spec.add_hfield(name="cobbles", ...)  # elevation 取 h, 方向按 meta 对齐
'''

# --------------------------------------------------------------------------------------
# 一份完整可跑的「静态地形」构建脚本（Isaac Sim 直调 omni.physx，作为兜底）
# --------------------------------------------------------------------------------------
ISAAC_RAW_SNIPPET = r'''
# 在 Isaac Sim 的 Script Editor 里可直接跑: 只建静态卵石地形, 用来跑大规模训练
import json, numpy as np
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd

BUNDLE = r"D:/pebble_scene/out_full"
P = PebbleBundle(BUNDLE)
stage = omni.usd.get_context().get_stage()
lay = P.layout(0)

root = UsdGeom.Xform.Define(stage, "/World/Cobbles")
for k in range(len(lay["pos"])):
    v, f = P.shape(int(lay["shape_id"][k]))
    path = f"/World/Cobbles/r{k:05d}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in (v * lay["scale"][k])])
    mesh.CreateFaceVertexCountsAttr([3] * len(f))
    mesh.CreateFaceVertexIndicesAttr(f.ravel().tolist())
    xf = UsdGeom.Xformable(mesh)
    xf.AddTranslateOp().Set(Gf.Vec3d(*lay["pos"][k]))
    q = lay["quat"][k]                             # wxyz -> USD 是 (real, i, j, k) = wxyz
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
'''


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "out_smoke"
    P = PebbleBundle(root)
    print("manifest:", json.dumps({k: P.manifest[k] for k in
                                   ("bundle_version", "units", "up_axis", "quat_order")}))
    print("shapes:", P.manifest["shapes"]["count"],
          "layouts:", P.manifest["layouts"]["n"],
          "pebbles/layout:", P.manifest["layouts"]["n_per_layout"][:6])
    lay = P.layout(0)
    print("layout0: n=%d fill_z=%.4f mass=%.1f..%.1f g"
          % (len(lay["pos"]), lay["fill_z"], 1e3 * lay["mass"].min(), 1e3 * lay["mass"].max()))
    print("stage boxes:", [b["name"] for b in P.stage_boxes])
    H, meta = P.heightfield(0)
    print("heightfield:", H.shape, H.dtype, "%.4f..%.4f m" % (H.min(), H.max()), meta["order"])
    print("subset y in [-1,0]:", len(P.subset(0, y=(-1.0, 0.0))), "pebbles")
