# -*- coding: utf-8 -*-
"""bench_mujoco.py —— 量 MuJoCo 里 N 颗鹅卵石的真实代价（编译时间 + 每步耗时）。

用法: python bench_mujoco.py <pebbles.xml> [wrapper.xml] [--steps=400]
不给 wrapper 时临时拼一份最小场景：地面 + 四周挡墙 + 我们的卵石 + 两个 0.18 m 球从 0.45 m
掉下来（近似"脚踩进石头里"的冲击，同时能看出床会不会被砸穿到地面）。
"""
import os
import sys
import tempfile
import time

import mujoco

pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
steps, dt, iters, solver = 400, 0.002, 50, "Newton"
for a in sys.argv[1:]:
    if a.startswith("--steps="):
        steps = int(a.split("=")[1])
    elif a.startswith("--dt="):
        dt = float(a.split("=")[1])
    elif a.startswith("--iters="):
        iters = int(a.split("=")[1])
    elif a.startswith("--solver="):
        solver = a.split("=")[1]

pebbles = os.path.abspath(pos_args[0]).replace("\\", "/")
assets = pebbles.replace(".xml", "_assets.xml")
wrapper = os.path.abspath(pos_args[1]).replace("\\", "/") if len(pos_args) > 1 else None

t0 = time.time()
if wrapper:
    print("[bench] wrapper: %s" % wrapper)
    model = mujoco.MjModel.from_xml_path(wrapper)
else:
    tmpl = """<mujoco>
  <option timestep="%(dt)s" iterations="%(iters)d" solver="%(solver)s"/>
  <include file="%(assets)s"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0"/>
    <geom name="wall_x0" type="box" size="0.04 1.42 0.15" pos="-1.50 0 0.15"/>
    <geom name="wall_x1" type="box" size="0.04 1.42 0.15" pos="1.50 0 0.15"/>
    <geom name="wall_y0" type="box" size="1.50 0.04 0.15" pos="0 -1.50 0.15"/>
    <geom name="wall_y1" type="box" size="1.50 0.04 0.15" pos="0 1.50 0.15"/>
    <include file="%(pebbles)s"/>
    <body name="dropper" pos="-0.6 0 0.45"><freejoint/>
      <geom type="sphere" size="0.09" density="1500"/></body>
    <body name="dropper2" pos="0.6 0 0.45"><freejoint/>
      <geom type="sphere" size="0.09" density="1500"/></body>
  </worldbody>
</mujoco>""" % dict(assets=assets, pebbles=pebbles, dt=dt, iters=iters, solver=solver)
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8")
    f.write(tmpl)
    f.close()
    print("[bench] wrapper: %s" % f.name)
    model = mujoco.MjModel.from_xml_path(f.name)

t_compile = time.time() - t0
data = mujoco.MjData(model)
print("[bench] nbody=%d ngeom=%d nmesh=%d compile=%.2fs"
      % (model.nbody, model.ngeom, model.nmesh, t_compile))
nb_peb = max(model.nbody - 7, 0)
print("[bench] pebble bodies=%d  total mass=%.0f kg" % (nb_peb, model.body_mass.sum()))

mujoco.mj_forward(model, data)
t0 = time.time()
for _ in range(steps):
    mujoco.mj_step(model, data)
dt = (time.time() - t0) / steps
print("[bench] %.2f ms/step at dt=%.4f -> %.0f%% of realtime (1 env), ncon=%d"
      % (1000 * dt, model.opt.timestep, 100 * (model.opt.timestep / dt), data.ncon))
for name in ("dropper", "dropper2"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        print("[bench] %s final z = %.3f (dropped from 0.450)" % (name, data.xpos[bid][2]))
