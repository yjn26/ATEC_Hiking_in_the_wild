#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isaac_bench.py —— 鹅卵石场景体检 (D1 静态 / D2 动态)

严格 A/B: 两种模式创建**完全相同**的 4467 个凸包 prim (同一套烹饪),
唯一差别是 dynamic 模式额外加 RigidBodyAPI + MassAPI + 阻尼。

测量
  1) t_build            场景构建墙钟时间
  2) t_step_*           每物理步耗时 (warmup 后 500 步)
  3) 沉降稳定性 (dynamic) 采样卵石的 z 下沉量 / 速度 / 水平漂移
  4) 60 kg 冲击块       从石面上方 3 cm 落下, 测块下沉量 + 卵石被推挤量
  5) NaN / 爆炸检测     速度 > 100 m/s 视为爆炸

用法
  python isaac_bench.py --bundle out_full --mode static|dynamic --out r.json [--limit 200]
"""
import argparse
import json
import math
import os
import sys
import time

# ---- 参数(必须在 import isaac 之前解析, 避免 AppLauncher 吃掉 argv) ----
p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
p.add_argument("--mode", required=True, choices=["static", "dynamic"])
p.add_argument("--out", required=True)
p.add_argument("--dt", type=float, default=0.001)
p.add_argument("--limit", type=int, default=0, help="只用前 N 颗卵石(冒烟用)")
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--bench-steps", type=int, default=500)
p.add_argument("--drop-steps", type=int, default=800)
p.add_argument("--sample", type=int, default=300, help="读状态的采样卵石数")
p.add_argument("--no-drop", action="store_true")
p.add_argument("--no-fabric", action="store_true",
               help="disable fabric so physics writes back to USD (needed to read real poses)")
p.add_argument("--vram-sample", action="store_true",
               help="后台线程每 200 ms 采一次 nvidia-smi 显存, 记录本次进程的峰值/基線")
p.add_argument("--render", action="store_true",
               help="step(render=True): USD xform 只在渲染/同步阶段更新, 读真实位姿必须开")
args, _unknown = p.parse_known_args()

T0 = time.time()


def log(*a):
    print("[%7.1fs] %s" % (time.time() - T0, " ".join(str(x) for x in a)), flush=True)


log("args:", args.mode, "bundle=", args.bundle, "limit=", args.limit)

# ---------------------------------------------------------------- 显存采样
# 注意: torch.cuda.max_memory_allocated() 只统计 PyTorch 分配器, PhysX 的内存不计入,
# 之前记的 vram_used_mb=0.1 是错的。改用 nvidia-smi 外部采样。
_vram = {"peak": 0, "floor": 10 ** 9, "n": 0}
_vram_stop = None


if args.vram_sample:
    import subprocess
    import threading

    _vram_stop = threading.Event()

    def _vram_loop():
        while not _vram_stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.splitlines()
                v = int(out[0].strip())
                _vram["peak"] = max(_vram["peak"], v)
                _vram["floor"] = min(_vram["floor"], v)
                _vram["n"] += 1
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)

    threading.Thread(target=_vram_loop, daemon=True).start()
    log("vram sampler started")

# ---------------------------------------------------------------- bundle 读取
import numpy as np

B = args.bundle
with open(os.path.join(B, "manifest.json"), "r", encoding="utf-8") as fh:
    MAN = json.load(fh)
with open(os.path.join(B, "static", "stage_boxes.json"), "r", encoding="utf-8") as fh:
    BOXES = json.load(fh)
z = np.load(os.path.join(B, "shapes", "rock_pool.npz"))
POOL_V, POOL_F, POOL_OFF = z["verts"], z["faces"], z["offsets"]
z = np.load(os.path.join(B, "layouts.npz"))
NPL = z["n_per_layout"]
OFF = np.concatenate([[0], np.cumsum(NPL)])
S, E = OFF[0], OFF[1]
LP = z["pos"][S:E]
LQ = z["quat"][S:E]
LS = z["scale"][S:E]
LSID = z["shape_id"][S:E]
LMASS = z["mass"][S:E]
LMU = z["mu"][S:E]
FILL_Z = float(z["fill_z"][0])
if args.limit:
    LP, LQ, LS, LSID, LMASS, LMU = (a[: args.limit] for a in (LP, LQ, LS, LSID, LMASS, LMU))
NPB = len(LP)
log("pebbles to build:", NPB, " fill_z=", FILL_Z)


def pool_shape(i):
    v0, v1 = int(POOL_OFF[i]), int(POOL_OFF[i + 1])
    f = POOL_F
    m = (f >= v0).all(1) & (f < v1).all(1)
    return POOL_V[v0:v1], f[m] - v0


# ---------------------------------------------------------------- 启动 Isaac
log("launching Isaac Sim (headless)...")
from isaaclab.app import AppLauncher

launcher = AppLauncher({"headless": True, "enable_cameras": False, "anti_aliasing": 0})
sim_app = launcher.app
log("Isaac Sim up")

import torch
from pxr import Usd, Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt, PhysxSchema

import omni.usd
import omni.physx
from isaaclab.sim import SimulationCfg, SimulationContext

sim = SimulationContext(SimulationCfg(dt=args.dt, device="cuda:0",
                                     use_fabric=not args.no_fabric))
stage = omni.usd.get_context().get_stage()
PHYSX = omni.physx.get_physx_interface()
log("SimulationContext ready, dt=", args.dt)

# fabric 开启时 PhysX 只写 fabric, USD xform 永远停在初始值; 本机实测连
# /physics/updateToUsd + update_transformations() 都不生效 —— 这是之前所有
# "物理不积分" 读数的真正原因 (物理一直在正常积分)。
# 结论: 位姿必须走 PhysX 张量视图直读 (下文的 rbv.get_transforms())。


# ---------------------------------------------------------------- 建场景辅助
def phys_material(path, mu_s, mu_d, rest=0.0):
    m = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(float(rest))
    return m


MAT_MORTAR = phys_material("/World/Looks/mortar", 0.5, 0.45, 0.0)
MAT_PEBBLE = phys_material("/World/Looks/pebble", 0.55, 0.50, 0.05)
MAT_BLOCK = phys_material("/World/Looks/block", 0.9, 0.8, 0.0)


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def add_box(path, center, size, mat, dynamic=False, mass=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    prim = c.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if mass:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    bind(prim, mat)
    return prim


def add_mesh(path, verts, faces, pos, quat, scale, mat, dynamic=False, mass=None, mu=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces.ravel()]))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(False)
    xf = UsdGeom.Xformable(m)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))
    prim = m.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        ma = UsdPhysics.MassAPI.Apply(prim)
        ma.CreateMassAttr(float(mass))
        ma.CreateDensityAttr(0.0)
        pb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        pb.CreateLinearDampingAttr(0.15)
        pb.CreateAngularDampingAttr(0.5)
        pb.CreateMaxLinearVelocityAttr(50.0)
    bind(prim, mat)
    UsdGeom.Imageable(prim).MakeInvisible()      # 关掉视觉, 只留碰撞
    return prim


# ---------------------------------------------------------------- 构建
log("building scene (%s)..." % args.mode)
t_build0 = time.perf_counter()

root = UsdGeom.Xform.Define(stage, "/World/envs").GetPrim()
UsdGeom.Xform.Define(stage, "/World/envs/env0")

for b in BOXES:
    add_box("/World/envs/env0/Stage/" + b["name"], b["center"], b["size"], MAT_MORTAR)

dyn = args.mode == "dynamic"
paths = []
for k in range(NPB):
    v, f = pool_shape(int(LSID[k]))
    pp = "/World/envs/env0/Pebbles/r%05d" % k
    add_mesh(pp, v, f, LP[k], LQ[k], LS[k], MAT_PEBBLE,
             dynamic=dyn, mass=float(LMASS[k]), mu=float(LMU[k]))
    paths.append(pp)

t_build = time.perf_counter() - t_build0
log("prims created in %.2fs (%.2f ms/prim)" % (t_build, 1000 * t_build / max(NPB, 1)))

# 冲击块 (60 kg)
# 块底面离石顶 3 cm: 中心 = 石顶(p95) + 间隙 3 cm + 半高 9 cm
BLOCK = "/World/envs/env0/ImpactBlock"
BLOCK_HALF_H = 0.09
BLOCK_GAP = 0.03
rock_top_p95 = float(np.percentile(LP[:, 2], 95))
block_z0 = rock_top_p95 + BLOCK_GAP + BLOCK_HALF_H
add_box(BLOCK, [0.0, 0.0, block_z0], [0.50, 0.30, 2 * BLOCK_HALF_H], MAT_BLOCK, dynamic=True, mass=60.0)
log("impact block: rock_top_p95=%.4f gap=%.2f -> center z=%.4f (bottom %.4f)"
    % (rock_top_p95, BLOCK_GAP, block_z0, block_z0 - BLOCK_HALF_H))


def sync():
    torch.cuda.synchronize()


log("sim.reset() ...")
t_reset0 = time.perf_counter()
sim.reset()
sync()
t_reset = time.perf_counter() - t_reset0
log("reset done in %.2fs (incl. PhysX cooking)" % t_reset)


# ---------------------------------------------------------------- 状态读取
# 必须走 PhysX 张量视图: 任何 USD / buffered 读数在本机都是陈旧值。
from isaacsim.core.simulation_manager import SimulationManager

sim_view = SimulationManager.get_physics_sim_view()
HAVE_VIEW = False
rbv = None
rbv_block = None
if dyn:
    rbv = sim_view.create_rigid_body_view("/World/envs/env0/Pebbles/r*")
    if getattr(rbv, "_backend", None) is None:
        raise RuntimeError("PhysX RigidBodyView backend is None —— PhysX 未创建刚体")
    view_paths = list(rbv.prim_paths)
    path_to_local = {p: k for k, p in enumerate(paths)}
    order = np.array([path_to_local[p] for p in view_paths], dtype=np.int64)
    rbv_block = sim_view.create_rigid_body_view(BLOCK)
    HAVE_VIEW = True
    log("RigidBodyView: %d bodies (built %d), order mapped OK" % (len(view_paths), NPB))
else:
    log("static mode: 无刚体, 跳过沉降/冲击回读")


# 采样只影响统计口径, 全量读很快, 因此统一读全量后再切片
idxs = np.arange(NPB) if NPB <= args.sample else np.sort(
    np.random.default_rng(0).choice(NPB, args.sample, replace=False))


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def read_pebbles():
    """返回 (NPB,3) 世界坐标, 顺序与构建顺序一致。"""
    t = _np(rbv.get_transforms())[:, 0:3].astype(np.float32)
    out = np.empty((NPB, 3), np.float32)
    out[order] = t
    return out


def block_pos():
    return _np(rbv_block.get_transforms())[0, 0:3].astype(float)


# ---------------------------------------------------------------- 步进
def run(n, render=None):
    rnd = args.render if render is None else render
    t0 = time.perf_counter()
    for _ in range(n):
        sim.step(render=rnd)
    sync()
    return time.perf_counter() - t0


log("warmup %d steps ..." % args.warmup)
run(args.warmup)

p0 = read_pebbles() if HAVE_VIEW else None
b0 = block_pos() if HAVE_VIEW else None

log("bench %d steps ..." % args.bench_steps)
t_bench = run(args.bench_steps)
p1 = read_pebbles() if HAVE_VIEW else None
ms_per_step = 1000.0 * t_bench / args.bench_steps

# 沉降期指标
if HAVE_VIEW:
    dz = p1[:, 2] - p0[:, 2]
    v = np.linalg.norm(p1 - p0, axis=1) / (args.bench_steps * args.dt)
    settle = dict(
        sink_max_mm=float(max(0.0, -dz.min()) * 1000),
        sink_mean_mm=float(-dz.mean() * 1000),
        drift_xy_max_mm=float(np.linalg.norm((p1 - p0)[:, :2], axis=1).max() * 1000),
        speed_mean_mps=float(v.mean()),
        speed_max_mps=float(v.max()),
        n_nan=int(np.isnan(p1).any(1).sum()),
    )
else:
    settle = None

# ---------------------------------------------------------------- 冲击试验
drop = None
if not args.no_drop and HAVE_VIEW:
    # 关键: 块必须在 drop 窗口开始时才放下。否则它在 warmup 阶段就已接触并落定,
    # drop 窗口测到的就是"静止"。这里用张量视图把块强行归位并清速。
    log("reset impact block -> z=%.4f (bottom %.4f), zero velocity" % (block_z0, block_z0 - BLOCK_HALF_H))
    dev = sim.device
    idx0 = torch.tensor([0], dtype=torch.int32, device=dev)
    # set_transforms 的位姿是 (pos3, quat_xyzw), 单位旋转 = (0,0,0,1)
    rbv_block.set_transforms(
        torch.tensor([[0.0, 0.0, block_z0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=dev),
        indices=idx0)
    rbv_block.set_velocities(torch.zeros((1, 6), dtype=torch.float32, device=dev), indices=idx0)
    run(2)  # 让归位生效
    log("drop 60 kg block, %d steps ..." % args.drop_steps)
    pd0 = read_pebbles()
    bd0 = block_pos()
    t_drop = run(args.drop_steps)
    pd1 = read_pebbles()
    bd1 = block_pos()
    dd = pd1 - pd0
    vmax = np.linalg.norm(dd, axis=1) / (args.drop_steps * args.dt)
    pen = (rock_top_p95 - (float(bd1[2]) - BLOCK_HALF_H)) * 1000.0
    drop = dict(
        rock_top_p95_m=rock_top_p95,
        block_gap_m=BLOCK_GAP,
        block_z_start=float(bd0[2]),
        block_z_end=float(bd1[2]),
        block_bottom_end_m=float(bd1[2]) - BLOCK_HALF_H,
        block_sink_mm=float((bd0[2] - bd1[2]) * 1000),
        penetration_below_rock_top_mm=float(pen),
        peb_count_moved_gt2mm=int((np.linalg.norm(dd, axis=1) > 0.002).sum()),
        peb_push_max_mm=float(np.linalg.norm(dd, axis=1).max() * 1000),
        peb_avg_speed_max_mps=float(vmax.max()),
        exploded=bool(vmax.max() > 100.0),
        ms_per_step_drop=float(1000.0 * t_drop / args.drop_steps),
    )
    if abs(drop["block_sink_mm"]) < 1e-6 and drop["peb_push_max_mm"] < 1e-6:
        log("WARNING: 完全没有运动 —— 检查 rbv 回读是否正确")

def gpu_vram():
    """进程内 nvidia-smi 采样结果; 若已停采样则含最终值"""
    if _vram_stop is not None:
        _vram_stop.set()
        time.sleep(0.25)
    floor = _vram["floor"] if _vram["n"] else None
    return dict(peak_mb=_vram["peak"] or None, floor_mb=floor, samples=_vram["n"])


res = dict(
    mode=args.mode, dt=args.dt, n_pebbles=NPB,
    readback="PhysX RigidBodyView.get_transforms()",
    t_build_s=round(t_build, 3), t_reset_s=round(t_reset, 3),
    t_total_s=round(time.time() - T0, 1),
    ms_per_step=round(ms_per_step, 4),
    steps_per_s=round(1000.0 / ms_per_step, 1) if ms_per_step > 0 else None,
    settle=settle, drop=drop,
    gpu=torch.cuda.get_device_name(0),
    torch_alloc_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
    vram=gpu_vram() if args.vram_sample else None,
)
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=True)
log("RESULT", json.dumps(res))
log("wrote", args.out)

# headless: sim_app.close() can hang; hard-exit right after writing results
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
