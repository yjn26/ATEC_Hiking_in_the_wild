#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pebble_loader.py —— 把鹅卵石 bundle 装进 Isaac Sim 场景（只建场，不含机器人）

设计原则
  1. 只依赖 `numpy` + `pxr`(USD)，**不 import isaaclab** → 与 isaaclab 0.47/0.54
     各版本、Isaac Sim 5.x 都可直接用。
  2. 只负责"建场"：建完返回句柄字典，不做 reset/step，不含机器人、观测、奖励。
  3. 位姿回读单独一个函数 `make_pebble_view()`，因为那是 Isaac Sim 的运行期 API，
     且**必须**走 PhysX 张量视图（USD / IsaacLab 的 data.* 在 fabric 下都是陈旧值）。

在一个已启动的 Kit/Isaac Sim python 环境里：

    from pebble_loader import build_pebble_scene, make_pebble_view

    sc = build_pebble_scene("/data/pebble_bundle", n_envs=1, layout_id=0, mode="dynamic")
    print(sc["n_pebbles"], sc["spawn"], sc["fill_z"])

    view = make_pebble_view(sc)              # 可选，需要在 sim.reset() 之后调用
    pos  = view.get_transforms()[:, 0:3]     # (N,3) 世界坐标

⚠️ 与 IsaacLab `InteractiveScene` 的配合
   IsaacLab 会用 env_0 的 prim 模板**自动克隆**到所有 env。本 loader 默认逐 env 显式
   建 prim（`n_envs` 份）。两者只能选一个，否则会重复：
     - 用 InteractiveScene 自动克隆  → 传 n_envs=1，只建 env_0；
     - 自己管理多 env              → 传 n_envs=N，不要给 scene 配自动克隆。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# ------------------------------------------------------------------ 物理常量
PHYS = dict(
    mu_mortar=(0.50, 0.45, 0.0),      # (static, dynamic, restitution) 槽体/平台
    mu_pebble=(0.55, 0.50, 0.05),     # 默认值; 逐颗真值在 layouts.npz 的 mu 字段
    damping_linear=0.15,
    damping_angular=0.5,
    max_linear_velocity=50.0,
    convex_approx="convexHull",
)

#: 槽体各 box 的物理材质分配（名字 -> PHYS 的键）
BOX_MATERIAL = {"default": "mu_mortar"}

#: 推荐物理参数（写进 HANDOFF，不在这里硬设，避免替使用者决定物理）
RECOMMENDED = dict(
    dt_s=1e-3,
    contact_offset_m=1e-3,
    rest_offset_m=5e-4,
    position_iterations=8,
    velocity_iterations=1,
)


# ------------------------------------------------------------------ bundle 读取
@dataclass
class BundleInfo:
    path: str
    manifest: dict
    n_layouts: int
    n_per_layout: np.ndarray
    pos: np.ndarray          # (N,3)
    quat_wxyz: np.ndarray    # (N,4)  **wxyz**（manifest 的 quat_order 字段保证）
    scale: np.ndarray        # (N,3)
    shape_id: np.ndarray     # (N,)
    mass: np.ndarray         # (N,)
    mu: np.ndarray           # (N,)
    fill_z: float
    pool_verts: np.ndarray
    pool_faces: np.ndarray
    pool_offsets: np.ndarray
    boxes: List[dict] = field(default_factory=list)
    heightfield: Optional[np.ndarray] = None
    heightfield_meta: Optional[dict] = None

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.n_per_layout)])


def load_bundle(path: str, with_heightfield: bool = False, layout_for_hf: int = 0) -> BundleInfo:
    """读 bundle 元数据 + 布局。纯 numpy，不依赖任何引擎 API。"""
    with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("quat_order") != "wxyz":
        raise ValueError("unexpected quat_order: %r" % (man.get("quat_order"),))
    if man.get("units") != "m" or man.get("up_axis") != "Z":
        raise ValueError("bundle 单位/上轴不符合预期: %r / %r" % (man.get("units"), man.get("up_axis")))

    z = np.load(os.path.join(path, "layouts.npz"))
    pool = np.load(os.path.join(path, "shapes", "rock_pool.npz"))
    with open(os.path.join(path, "static", "stage_boxes.json"), "r", encoding="utf-8") as fh:
        boxes = json.load(fh)

    hf = hf_meta = None
    if with_heightfield:
        hf_path = os.path.join(path, "static", "heightfield_L%02d.npy" % layout_for_hf)
        if os.path.exists(hf_path):
            hf = np.load(hf_path)
            meta_path = os.path.join(path, "static", "heightfield_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as fh:
                    hf_meta = json.load(fh)

    return BundleInfo(
        path=path, manifest=man,
        n_layouts=int(len(z["n_per_layout"])),
        n_per_layout=z["n_per_layout"].astype(np.int64),
        pos=z["pos"], quat_wxyz=z["quat"], scale=z["scale"], shape_id=z["shape_id"],
        mass=z["mass"], mu=z["mu"], fill_z=float(z["fill_z"][0]),
        pool_verts=pool["verts"], pool_faces=pool["faces"], pool_offsets=pool["offsets"],
        boxes=boxes, heightfield=hf, heightfield_meta=hf_meta,
    )


def layout_indices(info: BundleInfo, layout_id: Union[int, str, Sequence[int]], n_envs: int,
                   seed: int = 0) -> List[int]:
    """把 layout_id 解析成每个 env 用哪个布局。支持 int / 'random' / 序列。"""
    off = info.offsets
    if isinstance(layout_id, str):
        if layout_id != "random":
            raise ValueError("layout_id 只能是 int / 'random' / 序列")
        rng = np.random.default_rng(seed)
        return [int(v) for v in rng.integers(0, info.n_layouts, size=n_envs)]
    if isinstance(layout_id, (list, tuple, np.ndarray)):
        seq = [int(v) for v in layout_id]
        if len(seq) != n_envs:
            raise ValueError("layout 序列长度 %d != n_envs %d" % (len(seq), n_envs))
        return seq
    return [int(layout_id)] * n_envs


def slice_layout(info: BundleInfo, li: int, limit: int = 0):
    off = info.offsets
    s, e = int(off[li]), int(off[li + 1])
    e = e if not limit else min(e, s + limit)
    sl = slice(s, e)
    return dict(
        pos=info.pos[sl], quat=info.quat_wxyz[sl], scale=info.scale[sl],
        shape_id=info.shape_id[sl], mass=info.mass[sl], mu=info.mu[sl],
        layout_index=li, global_offset=s, count=e - s,
    )


# ------------------------------------------------------------------ USD 建场
def _shape_mesh(info: BundleInfo, sid: int):
    a, b = int(info.pool_offsets[sid]), int(info.pool_offsets[sid + 1])
    verts = info.pool_verts[a:b]
    faces = info.pool_faces
    keep = (faces >= a).all(1) & (faces < b).all(1)
    return verts, faces[keep] - a


def build_pebble_scene(bundle: Union[str, BundleInfo], n_envs: int = 1,
                       layout_id: Union[int, str, Sequence[int]] = 0, mode: str = "dynamic",
                       prim_root: str = "/World/envs", with_boxes: bool = True,
                       with_pebbles: bool = True, make_invisible: bool = True,
                       limit: int = 0, spawn_clearance: float = 0.005,
                       env_spacing: float = 4.0,
                       contact_offset: Optional[float] = None,
                       rest_offset: Optional[float] = None,
                       verbose: bool = True, **_) -> Dict:
    """在当前打开的 USD stage 上建场。返回句柄字典。

    mode: "dynamic" 加 RigidBodyAPI+MassAPI（可被踩动）；"static" 只做碰撞体。

    contact_offset / rest_offset: PhysX 接触带。默认 None = 引擎默认（2 cm），
    与卵石尺寸同阶，会让整床卵石表现得像一块刚性板；想看到“踩下去会陷”
    必须显式调到 1–2 mm（见 RECOMMENDED）。同时也要约束机器人脚的
    contact offset，因为一对碰撞体的接触距离是两侧之和。
    """
    if mode not in ("dynamic", "static"):
        raise ValueError("mode 必须是 dynamic / static")
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt, PhysxSchema

    info = bundle if isinstance(bundle, BundleInfo) else load_bundle(bundle)
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    dyn = mode == "dynamic"

    # ---- 物理材质 ----
    def _mat(path, mu_s, mu_d, rest):
        m = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(float(mu_s))
        api.CreateDynamicFrictionAttr(float(mu_d))
        api.CreateRestitutionAttr(float(rest))
        return m

    MAT_MORTAR = _mat("/World/Looks/pebble_mortar_loader", *PHYS["mu_mortar"])
    MAT_PEBBLE = _mat("/World/Looks/pebble_loader", *PHYS["mu_pebble"])

    def _bind(prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    out = dict(n_envs=n_envs, mode=mode, fill_z=info.fill_z, prim_root=prim_root,
               box_paths=[], pebble_paths=[], pebbles_per_env=[], layout_ids=[],
               _limit=limit,
               recommended=dict(RECOMMENDED))

    # 显式建父 xform；多 env 必须在 X 上拉开间距，否则两份场景完全重叠、
    # 卵石两两互穿，接触会把它们推开（实测 7 mm）。
    UsdGeom.Xform.Define(stage, prim_root)
    out["env_spacing"] = float(env_spacing)
    out["env_origins"] = [(k * float(env_spacing), 0.0, 0.0) for k in range(n_envs)]
    for k in range(n_envs):
        ex = UsdGeom.Xform.Define(stage, "%s/env_%d" % (prim_root, k))
        ex.AddTranslateOp().Set(Gf.Vec3d(k * float(env_spacing), 0.0, 0.0))

    # ---- 槽体 / 平台 / 站立台 / 填充体 ----
    if with_boxes:
        for k in range(n_envs):
            for b in info.boxes:
                p = "%s/env_%d/Stage/%s" % (prim_root, k, b["name"])
                c = UsdGeom.Cube.Define(stage, p)
                c.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(c)
                xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in b["center"]]))
                xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in b["size"]]))
                prim = c.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim)
                _apply_contact_offsets(prim, contact_offset, rest_offset)
                _bind(prim, MAT_MORTAR)
                out["box_paths"].append(p)

    # ---- 卵石 ----
    if with_pebbles:
        _mesh_cache = {}
        lids = layout_indices(info, layout_id, n_envs)
        out["layout_ids"] = lids
        for k in range(n_envs):
            L = slice_layout(info, lids[k], limit)
            n = L["count"]
            out["pebbles_per_env"].append(n)
            for i in range(n):
                sid = int(L["shape_id"][i])
                if sid not in _mesh_cache:
                    _mesh_cache[sid] = _shape_mesh(info, sid)
                v, f = _mesh_cache[sid]
                p = "%s/env_%d/Pebbles/r%05d" % (prim_root, k, i)
                m = UsdGeom.Mesh.Define(stage, p)
                m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(a), float(b_), float(c)) for a, b_, c in v]))
                m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
                m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(t) for t in f.ravel()]))
                m.CreateSubdivisionSchemeAttr("none")
                m.CreateDoubleSidedAttr(False)
                xf = UsdGeom.Xformable(m)
                xf.AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in L["pos"][i]]))
                q = L["quat"][i]
                # USD Gf.Quatf(real, i, j, k) —— bundle 里就是 wxyz，直接对上
                xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
                xf.AddScaleOp().Set(Gf.Vec3f(*[float(x) for x in L["scale"][i]]))
                prim = m.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(PHYS["convex_approx"])
                _apply_contact_offsets(prim, contact_offset, rest_offset)
                if dyn:
                    UsdPhysics.RigidBodyAPI.Apply(prim)
                    ma = UsdPhysics.MassAPI.Apply(prim)
                    ma.CreateMassAttr(float(L["mass"][i]))   # 显式质量，别让引擎按尺寸猜
                    ma.CreateDensityAttr(0.0)
                    pb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    pb.CreateLinearDampingAttr(PHYS["damping_linear"])
                    pb.CreateAngularDampingAttr(PHYS["damping_angular"])
                    pb.CreateMaxLinearVelocityAttr(PHYS["max_linear_velocity"])
                _bind(prim, MAT_PEBBLE)
                if make_invisible:
                    UsdGeom.Imageable(prim).MakeInvisible()
                out["pebble_paths"].append(p)

    # ---- spawn 锚点（站立台顶面 + 余量），朝向 +Y；**env 局部坐标** ----
    out["spawn"], out["spawn_yaw_deg"] = derive_spawn(info.boxes, spawn_clearance)
    out["n_pebbles"] = len(out["pebble_paths"])
    if verbose:
        print("[pebble_loader] envs=%d spacing=%.2fm mode=%s pebbles=%d boxes=%d fill_z=%.5f spawn=%s"
              % (n_envs, env_spacing, mode, out["n_pebbles"], len(out["box_paths"]), info.fill_z, out["spawn"]),
              flush=True)
    return out


def _apply_contact_offsets(prim, contact_offset: Optional[float], rest_offset: Optional[float]) -> None:
    """给碰撞体写 PhysX 接触带。两个都为 None 时什么都不做（保留引擎默认）。

    PhysX 默认 contactOffset = 0.02 m，与卵石（直径 ~2 cm）同阶：接触在相距 2 cm
    时就生成，整床卵石因此表现为一块刚性板，机器人踩上去不会下沉、卵石不会被
    推开。一对碰撞体的接触距离是**两侧之和**，所以只改卵石侧不够，机器人脚的
    contact offset 也得同等约束（见 play.py 的 --pebble_contact_offset）。
    """
    if contact_offset is None and rest_offset is None:
        return
    from pxr import PhysxSchema

    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    if contact_offset is not None:
        api.CreateContactOffsetAttr(float(contact_offset))
    if rest_offset is not None:
        api.CreateRestOffsetAttr(float(rest_offset))


def derive_spawn(boxes: List[dict], clearance: float = 0.005):
    """从 standing_pad 推出出生点：顶面中心 + 余量；朝向 +Y（yaw=90°）。"""
    for b in boxes:
        if "stand" in b["name"]:
            c, s = b["center"], b["size"]
            return (float(c[0]), float(c[1]), float(c[2]) + float(s[2]) / 2.0 + clearance), 90.0
    raise ValueError("stage_boxes.json 里找不到 standing_pad")


# ------------------------------------------------------------------ 运行期回读
def make_pebble_view(scene: Dict, device: str = "cuda:0"):
    """在 sim.reset() 之后调用，返回 PhysX RigidBodyView（世界位姿直读）。

    多版本兼容：优先 isaacsim.core.simulation_manager，退化到 omni.physics.tensors。
    """
    root = scene["prim_root"]
    n_envs = scene["n_envs"]
    pattern = ("%s/env_*/Pebbles/r*" % root) if n_envs > 1 else ("%s/env_0/Pebbles/r*" % root)
    try:
        from isaacsim.core.simulation_manager import SimulationManager

        sv = SimulationManager.get_physics_sim_view()
        return sv.create_rigid_body_view(pattern)
    except Exception:  # noqa: BLE001
        pass
    for mod_name in ("omni.physics.tensors", "omni.physx.tensors"):
        try:
            mod = __import__(mod_name, fromlist=["create_rigid_body_view"])
            return mod.create_rigid_body_view(pattern)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("无法创建 PhysX 刚体视图（Isaac Sim 版本不匹配？）")


def view_positions(view) -> np.ndarray:
    """(N,3) 世界坐标。注意：这是 PhysX 直读，比 USD / data.* 可靠。"""
    t = view.get_transforms()
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return t[:, 0:3]


def wake_mask(robot_xy: np.ndarray, pebble_xy: np.ndarray, radius: float) -> np.ndarray:
    """唤醒掩码：距离机器人 base 的 XY 平面半径内的卵石。纯 numpy，可直接用于训练循环。

    本场景实测密度 745 颗/m²；半径 1.5 m 的圆盘(7.07 m²)比整个槽(6 m²)还大，
    等于全开。有效区间是 0.4~0.7 m。
    """
    d2 = ((pebble_xy[:, 0] - robot_xy[0]) ** 2 + (pebble_xy[:, 1] - robot_xy[1]) ** 2)
    return d2 <= radius * radius


# ------------------------------------------------------------------ 自检
def verify_scene(scene: Dict, info: BundleInfo, tol: float = 1e-6) -> Dict:
    """可判定断言：颗数、prim 存在性、位姿与 npz 一致、填充体顶面 = fill_z。

    不依赖物理；直接读 USD（建场后 USD 就是真值）。
    """
    from pxr import Usd, UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    rep = dict(n_pebbles=scene["n_pebbles"], missing_prim=0, max_pos_err=0.0, max_scale_err=0.0,
               fill_ok=None, ok=True)

    expect = []
    origins = scene.get("env_origins") or [(0.0, 0.0, 0.0)] * len(scene["layout_ids"])
    for k, li in enumerate(scene["layout_ids"]):
        L = slice_layout(info, li, scene.get("_limit", 0))
        off = np.asarray(origins[k], dtype=np.float64)[None, :]
        expect.append(L["pos"].astype(np.float64) + off)
    E = np.concatenate(expect, 0) if expect else np.zeros((0, 3))

    if scene["n_pebbles"] != len(E):
        rep["ok"] = False
        raise AssertionError("颗数不符: built=%d expect=%d" % (scene["n_pebbles"], len(E)))

    tc = Usd.TimeCode.Default()
    for i, p in enumerate(scene["pebble_paths"]):
        prim = stage.GetPrimAtPath(p)
        if not prim:
            rep["missing_prim"] += 1
            continue
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc).ExtractTranslation()
        err = np.abs(np.array([t[0], t[1], t[2]]) - E[i]).max()
        rep["max_pos_err"] = max(rep["max_pos_err"], float(err))

    if rep["missing_prim"]:
        rep["ok"] = False

    for b in info.boxes:
        if b["name"] != "bulk_fill":
            continue
        top = float(b["center"][2]) + float(b["size"][2]) / 2.0
        rep["fill_top_m"] = round(top, 6)
        rep["fill_ok"] = abs(top - info.fill_z) < 1e-4
    if not rep["fill_ok"]:
        rep["ok"] = False

    if rep["max_pos_err"] > tol:
        rep["ok"] = False
    return rep
