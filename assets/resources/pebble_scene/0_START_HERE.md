# 先看这一页（60 秒）

## 这是什么

一个**鹅卵石测试地形**：1.5 × 4 m 的卵石槽（槽深 20 cm）+ 边上高出石顶约 10 cm 的平台 + 站立台。
数据是引擎无关的裸数字，用 `pebble_loader.py` 一行建成 Isaac 场景里的碰撞体/刚体。

## 这**不是**什么（重要，避免白干半天）

- ❌ **不是 env**。包里没有机器人、没有观测、没有奖励、没有 `xxx_env_cfg.py`。
- ❌ **不能直接训练**。把它接进你们现有的 env，只需要加一行函数调用（见下）。
- ❌ **不是 USD 资产**。是 `layouts.npz`（3.14 MB）+ 一个建场函数，在目标机几秒重建。

## 30 秒验收

```bash
# 包根目录下（PY 指向能跑 Isaac Sim 的 python）
PY=/path/to/python bash run_selftest.sh
```

看到 `SELFTEST OK (all)` 就是接好了。日志在 `bench_out/`。
（判据是输出里出现 `SELFTEST OK`——Isaac 会把异常吞成 exit 0，不能只看返回码。）

## 三步接入你们的 env

```python
from pebble_loader import build_pebble_scene, make_pebble_view

sc = build_pebble_scene("bundle", n_envs=1, layout_id=0, mode="dynamic")
# sc: n_pebbles / prim_paths / spawn / spawn_yaw_deg / fill_z / env_origins
view = make_pebble_view(sc)            # 必须在 sim.reset() 之后
pos  = view.get_transforms()[:, 0:3]   # (N,3) 世界坐标
```

在 `InteractiveScene` 里两种接法**二选一**（否则重复建两遍）：
- 靠 IsaacLab 从 env_0 自动克隆 → 传 `n_envs=1`
- 自己管多 env → 传 `n_envs=N`，并**给 scene 关掉自动克隆**

## 版本矩阵

| 组件 | 要求 | 说明 |
|---|---|---|
| `pebble_loader.py` | numpy；Kit 里的 `pxr` / `omni.usd` | **不 import isaaclab**，0.47 / 0.54.x 都能用 |
| `make_pebble_view()` | Isaac Sim 5.x | 优先 `isaacsim.core.simulation_manager`，回退 `omni.physics.tensors` |
| `loader_selftest.py` | isaaclab 0.54.x（用其 `AppLauncher`） | 若你们只有 Isaac Sim 没有 IsaacLab，自检要改成 `from isaacsim import SimulationApp` |
| 参考实测 | Isaac Sim 5.1 + IsaacLab 0.54.3 + RTX 5090 | 见 `physics_hints.json` |

## 两个要填的参数

| 参数 | 默认 | 什么时候改 |
|---|---|---|
| `spawn` | (1.40, −1.45, 0.26)，朝 +Y | 站立台是 `generic_large`（1.20×1.00 m）；换机型改 `bundle/layout_config.json` 的 `robot_class` 重生成 |
| 唤醒半径 | 0.6 m（≈840 颗/env） | 见 `HANDOFF.md` §3。**别用 1.5 m**——本密度下那比整个槽还大，等于全开 |

## 三个最容易踩的坑

1. **位姿只能走 `make_pebble_view()` 读**。USD 的 xform 在 fabric 下永不更新，IsaacLab 的
   `data.root_pos_w` 不调 `update()` 也不刷新——两条路都会给你一个"物理不积分"的假象。
   指纹：位姿恒定且 `vz` 恰好等于 `2·g·dt`。
2. **多 env 必须设 `env_spacing`**（默认 4 m）。为 0 时两份场景完全重叠、卵石互穿，
   接触会把它们推开约 7 mm。
3. **四元数**：bundle 是 `wxyz`；USD `Gf.Quatf` 直接对上；但 PhysX `set_transforms` 要 `xyzw`。

## 文件清单

```
0_START_HERE.md      本页
HANDOFF.md           七条契约 + 实测数据 + 坑（要动代码前请读）
pebble_loader.py     建场函数（主体）
loader_selftest.py   自检 A1~A6
run_selftest.sh      一键自检
physics_hints.json   实测性能/物理（仅供参考，换机器要重测）
bundle/              layouts.npz / rock_pool.npz / stage_boxes.json / heightfield_L*.npy
bundle/visual/       rock_pool.obj（32 个单位形状）/ preview.png
                     （cobble_L00.obj 合并网格 v3 起不再随包：44 MB 且可随时用
                      `python gen_pebbles.py --layouts 1 --static-mesh` 重建）
config_mujoco_basin.json  MuJoCo 侧（2.84x2.84 盆）用的配置
export_mujoco.py / bench_mujoco.py  导出 MuJoCo 资产 / 实测步速
bench/               原始日志（sweep / 自检 / 冲击与沉降）
```

> ⚠️ `physics_hints.json` 里的耗时、显存、沉降量是**那台机器的**（Isaac Sim 5.1 / RTX 5090 /
> powersave CPU）。换机器请重跑 `run_selftest.sh` 和一次冲击测试，只把量级当参考。
