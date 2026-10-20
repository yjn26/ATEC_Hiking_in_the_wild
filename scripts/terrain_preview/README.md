# Parkour 课程地形全量导出（terrain-only）

这里是 `scripts/pebble_terrain_preview.py` 的一次完整输出：**7 个子地形 × 10 档课程难度**，
逐 tile 导出 OBJ、山影色图、统计。

生成过程中**没有启动环境、没有机器人、没有 PPO**——只烘焙地形。Kit 只被用来 import
配置模块。

```bash
export PYTHONPATH=$PWD/source/instinctlab
python scripts/pebble_terrain_preview.py --headless --out /tmp/terrain_preview
# 只想看数据、不要 OBJ：
python scripts/pebble_terrain_preview.py --headless --out /tmp/tp --no-obj
```

单次全量：**烘焙 17.7 s**（200 tile，关墙 + 关 flat-patch 采样）+ 导出约 2 min（70 个 OBJ，126 MB）。

## 为什么必须走 `TerrainGenerator`

`TerrainGenerator.__init__` 会把每个子地形 cfg 的参数**改写**成课程级参数：

| 参数 | 子地形 cfg 默认值 | 训练实际值（`ROUGH_TERRAINS_CFG`） |
|---|---|---|
| `size` | (10, 10) m | **(8, 8) m** |
| `horizontal_scale` | 0.1 m | **0.05 m** |
| `vertical_scale` | 0.005 m | 0.005 m |
| `slope_threshold` | 1.0 | 1.0 |
| 难度 | — | 第 r 行取 `(r + U(0,1)) / num_rows` |

所以直接 `cfg.function(difficulty, cfg)` 拿到的是 **10 m tile / 0.1 m 网格 / 难度端点值** 的假地形
（本次修复前导出的 OBJ 就是这种：`x:[0,10]`、10201 顶点）。脚本改为实例化真正的
`TerrainGenerator`，逐 tile 取 `generator.terrain_meshes[col * num_rows + row]`，
再平移回 `0..size` 本地坐标系。

另外两个只在导出时生效的覆盖（不影响高度场本身）：

* `wall_prob = [0, 0, 0, 0]` —— 课程里每个 tile 四边各有 5 m 护栏墙（`wall_height=5.0`），
  纯边界几何。只把 `wall_height` 置 0 是不够的：`wall_prob>0` 仍会生成一圈**零高度面**，
  表现为 `x_span=8.05 m` 和一批假的 90° 面。
* `flat_patch_sampling = None` —— 只影响出生点采样，不碰网格；关掉后烘焙 22 s → 18 s。

## 产物

```
png/overview.png          80 m × 160 m 整场（200 tile），灰度山影 + 卵石 4 m 放大内嵌
png/<terrain>.png         每个子地形 10 档难度的山影色图（每格 8 m）
stats.csv                 逐 tile：n_tri / z_min / z_max / z_mean / z_p95 / slope_* / area_above_5mm
meta.json                 生成参数、列布局、逐 tile 统计（JSON）
```

列布局（`num_cols=20`，proportion 归一到 20 列）：

| 子地形 | proportion | 列数 | 列号 |
|---|---|---|---|
| perlin_rough | 0.15 | 3 | 0–2 |
| perlin_rough_stand | 0.10 | 2 | 3–4 |
| pyramid_stairs | 0.15 | 3 | 5–7 |
| pyramid_stairs_high | 0.10 | 2 | 8–9 |
| pyramid_stairs_inv | 0.15 | 3 | 10–12 |
| pyramid_stairs_inv_high | 0.10 | 2 | 13–14 |
| **pebble_bed** | **0.25** | **5** | **15–19** |

## 实测（本目录 `stats.csv`，行 0 / 4 / 9 = 难度最低 / 中 / 最高）

| 地形 | z 峰谷 L0 / L4 / L9 [mm] | 坡度 p95 L0 / L4 / L9 [°] | `area_above_5mm` L9 |
|---|---|---|---|
| perlin_rough | 0 / ±20 / ±45 | 0 / 12.6 / 24.1 | 0.30 |
| pebble_bed | [0, 60] / [0, 70] / [0, 85] （均值 22 → 32） | 24.1 / 26.6 / 27.0 | 0.995 |

卵石床（`PebbleBedTerrainCfg`：`stone_radius_range=(0.060, 0.130)`、`crest_height_ratio=(0.30, 0.45)`、
`size_difficulty_gain=0.45`、`crevice_fill=0.25`、28° 坡度钳制）：

* 石缝底面被 `crevice_fill` 填到 1 个垂直单位（5 mm），99.5% 的表面高于 5 mm，人走上去是"整片砾石面"而不是"石头间有坑"。（`stats.csv` 里 `z_min=0.000` 来自下面说的重复顶点 artifact，不是高度场本身。）
* 坡度 p95 24–27° < 28°，`slope_threshold=1.0`（45°）不会把卵石面拉成竖直面。
* `slope_p999` 在 L0/L4 是 39–45°：28° 钳制限制的是**轴向**相邻格高差
  (`tan28° × 0.05 m = 26.6 mm`)，斜对角三角形梯度可达 `atan(√2·tan28°) ≈ 37°`。
  若要让**三角面**也严格 ≤28°，把 `max_step` 乘 `1/√2` 即可（本次未改，保持与正在跑的训练一致）。
* 每个 tile 有 1–35 个退化面 / 60–124 个 90° 面（0.24% 的格子），来源是 IsaacLab
  `height_field_to_mesh` 阶段在少数格子上生成了 **(x,y) 重复顶点**（z=0 与真实高度并存）。
  `perlin_rough` 在关墙后为 0 个，说明它与卵石高度场的具体取值有关，不是本仓库代码引入；
  影响面 0.24% × 1.5 mm²，物理上可忽略，但会污染 `slope_max` 这一类极值统计。

## 5 cm 网格是精度天花板

真实砾石 2–6 cm 在 5 cm 网格下表达不了，所以生成的是 12–26 cm 块石、露出 3–7 cm。
想要 2.5 cm 网格得把课程 `horizontal_scale` 改成 0.025：三角面 ×4，烘焙 18 s → 60–80 s。

## 训练侧的静态高度场限制

训练地形只能是静态高度场：4096 env × 1.88 万颗卵石 ≈ 7700 万刚体，不可能；
`terrain_importer` 每 tile 只烘一张 trimesh。所以训练里学到的是"走不规则卵石路面"，
**脚推不开石头**。"踩开砾石"只能由 play 场景里的刚性卵石床（v3，18847 颗）体现。
