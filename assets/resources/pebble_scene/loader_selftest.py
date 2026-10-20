# -*- coding: utf-8 -*-
"""loader_selftest.py — pebble_loader 的可判定自检

用法:  python loader_selftest.py --bundle out_full --mode static|dynamic [--limit N]

断言（不通过就 raise / 非零退出）:
  A1 建出的 prim 颗数 == npz 颗数
  A2 每个 prim 的世界坐标与 npz 的 max|Δ| <= 1e-6
  A3 bulk_fill 顶面 == fill_z（1e-4）
  A4 spawn == 站立台顶面 + 余量
  A5 (dynamic) PhysX 刚体视图的 count 与颗数一致，且首/末位姿与 npz 对齐
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T0 = time.time()


def log(*a):
    print("[%6.1fs] %s" % (time.time() - T0, " ".join(str(x) for x in a)), flush=True)


p = argparse.ArgumentParser()
p.add_argument("--bundle", default="out_full")
p.add_argument("--mode", default="static", choices=["static", "dynamic"])
p.add_argument("--limit", type=int, default=0)
p.add_argument("--envs", type=int, default=1)
args, _unknown = p.parse_known_args()

from isaaclab.app import AppLauncher

launcher = AppLauncher({"headless": True, "enable_cameras": False, "anti_aliasing": 0})
app = launcher.app
log("Isaac Sim up")

import numpy as np
import torch

from isaaclab.sim import SimulationCfg, SimulationContext

import pebble_loader as PL

fails = []


def check(name, ok, detail=""):
    log("%-6s %-42s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        fails.append(name)


log("loading bundle ...")
info = PL.load_bundle(args.bundle, with_heightfield=True)
log("bundle: layouts=%d n_per_layout[0..3]=%s fill_z=%.6f quat_order=%s"
    % (info.n_layouts, info.n_per_layout[:4].tolist(), info.fill_z, info.manifest.get("quat_order")))

sim = SimulationContext(SimulationCfg(dt=PL.RECOMMENDED["dt_s"], device="cuda:0"))
log("SimulationContext dt=", sim.get_physics_dt())

t_build0 = time.perf_counter()
sc = PL.build_pebble_scene(info, n_envs=args.envs, layout_id=0, mode=args.mode, limit=args.limit)
t_build = time.perf_counter() - t_build0
log("built in %.2fs: pebbles=%d boxes=%d mode=%s" % (t_build, sc["n_pebbles"], len(sc["box_paths"]), sc["mode"]))
log("spawn=%s yaw=%.1f fill_z=%s" % (sc["spawn"], sc["spawn_yaw_deg"], sc["fill_z"]))

# ---------------- A1 颗粒数 ----------------
L0 = PL.slice_layout(info, 0, args.limit)
check("A1 prim count == npz count × envs",
      sc["n_pebbles"] == L0["count"] * args.envs and sc["n_pebbles"] == len(sc["pebble_paths"]),
      "built=%d expect=%d (%d×%d)" % (sc["n_pebbles"], L0["count"] * args.envs, L0["count"], args.envs))

# ---------------- A2 位姿一致 ----------------
rep = PL.verify_scene(sc, info, tol=1e-6)
check("A2 max|Δpos| <= 1e-6", rep["max_pos_err"] <= 1e-6, "max=%.3e missing=%d" % (rep["max_pos_err"], rep["missing_prim"]))
check("A3 fill top == fill_z", bool(rep["fill_ok"]), "top=%s fill_z=%.6f" % (rep.get("fill_top_m"), info.fill_z))

# ---------------- A4 spawn ----------------
exp_spawn = PL.derive_spawn(info.boxes, 0.005)[0]
check("A4 spawn == standing_pad top + 5mm", np.allclose(sc["spawn"], exp_spawn, atol=1e-9),
      "spawn=%s expect=%s" % (sc["spawn"], exp_spawn))

if args.mode == "dynamic":
    log("sim.reset() ...")
    sim.reset()
    try:
        view = PL.make_pebble_view(sc)
        n_view = getattr(view, "count", None)
        check("A5 view count == pebbles", (n_view == sc["n_pebbles"]), "view=%s built=%s (envs=%d)" % (n_view, sc["n_pebbles"], sc["n_envs"]))
        pos = PL.view_positions(view)
        # view 顺序按 prim 路径（env_0/r00000.. 然后 env_1/...），env_0 在最前面
        per_env = sc["n_pebbles"] // max(1, sc["n_envs"])
        d = np.linalg.norm(pos[:per_env] - L0["pos"], axis=1)
        # 注意：sim.reset() 后已经过了 2 步物理，床层会因已知的残余重叠（<=1.73 mm）
        # 被接触推开一点点。所以这里判的是「量级 + 分位」，而不是零误差。
        p50, p99, dmax = float(np.percentile(d, 50)), float(np.percentile(d, 99)), float(d.max())
        check("A6 view 位姿与 npz 对齐 (p99<=3mm, max<=5mm)", (p99 <= 3e-3 and dmax <= 5e-3),
              "p50=%.2fmm p99=%.2fmm max=%.2fmm (残余重叠上限 1.73mm)"
              % (p50 * 1e3, p99 * 1e3, dmax * 1e3))
        log("view z range: %.5f .. %.5f  (fill_z=%.5f)" % (pos[:, 2].min(), pos[:, 2].max(), info.fill_z))
    except Exception as exc:  # noqa: BLE001
        check("A5 make_pebble_view", False, repr(exc))

log("=" * 70)
if fails:
    log("SELFTEST FAILED:", fails)
    sys.stdout.flush()
    os._exit(1)
log("SELFTEST OK (%s, %d pebbles, build %.2fs)" % (args.mode, sc["n_pebbles"], t_build))
sys.stdout.flush()
os._exit(0)
