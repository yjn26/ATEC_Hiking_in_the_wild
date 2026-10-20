# PEBBLE SCENE — 交接说明（HANDOFF）

面向**机器人仿真团队**。本包只提供「地形 + 鹅卵石床」这一层，**不含**机器人、观测、奖励、训练脚本。

```
bundle/                 场景数据（3.17 MB 必需 / 28 MB 全量）
pebble_loader.py        建场函数（纯 pxr + numpy，零 isaaclab 依赖）
loader_selftest.py      可判定自检（A1~A6，不通过即非零退出）
physics_hints.json      已测物理/性能数据
HANDOFF.md              本文件
```

---

## 1. 三行接入

```python
from pebble_loader import build_pebble_scene, make_pebble_view

sc = build_pebble_scene(bundle="/data/pebble_bundle", n_envs=1, layout_id=0, mode="dynamic")
# sc: n_pebbles / prim_paths / spawn / spawn_yaw_deg / fill_z / layout_ids / recommended
view = make_pebble_view(sc)          # 必须在 sim.reset() 之后
pos  = view.get_transforms()[:, 0:3] # (N,3) 世界坐标
```

`pebble_loader.py` **不 import isaaclab**，只依赖 `numpy` + `pxr`(+`omni.usd`)。因此与
isaaclab 0.47 / 0.54.x、Isaac Sim 5.x 都能配。`make_pebble_view()` 里对
`isaacsim.core.simulation_manager` / `omni.physics.tensors` 做了多版本回退。

### 与 `InteractiveScene` 的两种接法（**二选一，否则重复**）

| 你的集成方式 | 传参 |
|---|---|
| IsaacLab 用 env_0 模板自动克隆 | `n_envs=1`，只建 env_0，让 scene 克隆 |
| 自己管理多 env | `n_envs=N`，且不要给 scene 配自动克隆 |

**多 env 间距很关键**：`env_spacing` 默认 **4.0 m**，env_k 整体沿 +X 平移
`k × env_spacing`。若不拉开，两份场景会完全重叠、卵石两两互穿，接触会把它们推开
（实测 7.35 mm，而单 env 只有 1.95 mm）。`sc["env_origins"]` 给出每个 env 的世界原点，
`sc["spawn"]` 是 **env 局部坐标**，要加 `env_origins[k]` 才是世界坐标。

---

## 2. 七条契约（**必须**对齐，错了会静默错）

1. **单位与朝向**：米制、**+Z 向上**、平台在 **+X 侧**、`spawn` 在
   `(1.40, −1.45, 0.255+0.005)`，朝 **+Y**（yaw = 90°）。
2. **四元数顺序 = `wxyz`**（`manifest.json` 里 `quat_order: "wxyz"` 显式声明）。
   USD 的 `Gf.Quatf(real, i, j, k)` 正好是 w-first，可直接喂。
   ⚠️ **PhysX 张量 API（`set_transforms`）要的是 `xyzw`**，转换错了会整体转 90°，且不报错。
3. **`scale` 是逐轴非均匀**。Isaac 里直接 `AddScaleOp`；**MuJoCo 的 mesh 不支持逐实例
   非均匀缩放**，必须烘进顶点或退化近似（见 §6）。
4. **`fill_z` 是不透明填充体（`bulk_fill`）顶面，不是地面**。v3（当前）四层包 `fill_z = 0.03000`
   （v2 两层 0.07500、v1 单层 0.10743），上面是 **4 层卵石**、18847 颗、床厚 **0.12 m**。
   `bulk_fill` 这块**不能省**——它挡住穿到槽底的接触，也是防穿透的兜底。
5. **质量与摩擦用 npz 里的真值，不要按尺寸猜**：质量均值 33.2 g（1.9~151 g），
   `mu` 逐颗 0.40~0.70。`loader` 已把 `MassAPI` 显式质量和 `density=0` 写好。
   （上列数字为 v1 单层包；v2 双层包同一口径，均值几乎不变。）
6. **prim 路径约定**：`/World/envs/env_<i>/Pebbles/r<index>`（5 位定宽，
   `r00000`…），槽体在 `/World/envs/env_<i>/Stage/<name>`。
   卵石采用 `r*` 通配，字典序与构建顺序一致。
7. **建议物理参数**（loader 不硬设，由你决定）：
   `dt ≤ 1 ms`（2 cm 石头 + 2 m/s 足速 → 每步位移 10% 尺度）、
   `contact_offset ≈ 1~2 mm`（**默认 2 cm 和石头一样大，等于裹了层果冻壳**）、
   `rest_offset ≈ 0.5 mm`、`position_iterations ≥ 8`。

---

## 3. 实测性能与显存（Isaac Sim 5.1 / RTX 5090 / 单 env）

| 例 | 颗数 | ms/步 | 步/s | 显存峰值 |
|---|---|---|---|---|
| 静态 | 4467 | **1.496** | 669 | 4875 MB |
| 动态 | 500 | 1.787 | 560 | 5210 MB |
| 动态 | 1000 | 2.056 | 486 | 5927 MB |
| 动态 | 2000 | 2.445 | 409 | 5993 MB |
| 动态 | 4467 | **2.423** | 413 | 6055 MB |

**两个反直觉的结论：**

- **动态成本在 ~2000 颗就饱和了**：4467 颗（2.423 ms）比 2000 颗（2.445 ms）还略快。
  瓶颈是固定开销（≈1.5 ms 静态基线 + ≈0.9 ms「场景里有动态刚体」），不是颗数。
  → **每 env 铺 500 还是 4500 颗，几乎不影响每步成本。**
- **显存才是真正的上限**：动态 4467 − 静态 4467 = **1180 MB / 4467 颗 ≈ 0.26 MB/颗**
  （粗估、偏保守，含不随规模增长的固定 GPU 池）。

按此推算（**外推，未实测**）：

| 配置 | 活跃刚体 | 估显存 | 判断 |
|---|---|---|---|
| 32 env × 800 | 25.6k | ~10 GB | 宽裕 |
| 64 env × 800 | 51.2k | ~16 GB | **32 GB 卡的安全区** |
| 128 env × 800 | 102k | ~29 GB | 危险 |
| 256 env × 1000 | 256k | ~70 GB | **不可行** |

**激活半径**（本场景实测密度 **745 颗/m²**）：

| 半径 | 面积 | 颗数 | 备注 |
|---|---|---|---|
| 0.4 m | 0.50 m² | ~375 | 偏紧（只够一只脚） |
| **0.6 m** | 1.13 m² | **~840** | **推荐默认** |
| 0.8 m | 2.01 m² | ~1500 | 保守 |
| ~~1.5 m~~ | 7.07 m² | ~5270 | **比整个槽（6 m²）还大 = 全开，等于没做** |

`pebble_loader.wake_mask(robot_xy, pebble_xy, radius)` 是现成的纯 numpy 实现。

---

## 4. 已验收项（`loader_selftest.py`）

跑过 3 个配置：**静态×1env / 动态×1env / 动态×2env**，全部通过。

| # | 断言 | 判据 | 实测 |
|---|---|---|---|
| A1 | 建出 prim 颗粒数 == npz 颗数 × envs | 完全相等 | 4467 / 4467 / 8934 |
| A2 | 每个 prim 世界坐标 vs npz | `max|Δ| ≤ 1e-6` | **0.000e+00** |
| A3 | `bulk_fill` 顶面 == `fill_z` | ≤ 1e-4 | 0.106274 |
| A4 | `spawn` == 站立台顶面 + 5 mm | `atol 1e-9` | (1.4, −1.45, 0.26) |
| A5 | PhysX 刚体视图 count == 颗数 | 完全相等 | 4467 / 8934 |
| A6 | 视图位姿 vs npz | p99 ≤ 3 mm, max ≤ 5 mm | p50 **0.06** / p99 **1.11** / max **1.95** mm |

> A6 不是零误差，这是**物理事实而非缺陷**：`sim.reset()` 之后已经过了 2 步物理，
> 床层因铺装验收时已知的**残余重叠（上限 1.73 mm）** 被接触推开最多 1.95 mm。
> 若看到几十厘米~米的偏差，那才是四元数/缩放/顺序搞错了。

复跑：
```bash
python loader_selftest.py --bundle out_full --mode dynamic            # 单 env
python loader_selftest.py --bundle out_full --mode dynamic --envs 2   # 多 env
```
构建耗时：4467 颗 **2.16 s**（0.48 ms/颗）；8934 颗（2 env）4.29 s。

### 物理行为（layout 0，dt = 1 ms，已在 Isaac 里实测）

| 项 | 数值 |
|---|---|
| 静止床层沉降 | 最大 **0.79 mm**、均值 0.003 mm、水平漂移 **1.91 mm**、NaN = 0 |
| 60 kg 块（0.50×0.30×0.18）从石顶上方 3 cm 落 | 下沉 **15.0 mm**，块底停在 0.1377 m |
| 穿透判据 | 高于石顶 p95 约 15 mm、高于填充体顶面 **30 mm**，**未穿透** |
| 被推挤卵石 | 24 颗 > 2 mm，最大位移 **38.7 mm**，无速度爆炸 |

→ 结论：**解析放置的床层在仿真里自洽，不需要额外做沉降。**

---

## 5. 你要填的两个参数

| 参数 | 现在填的是 | 什么时候改 |
|---|---|---|
| 站立台尺寸 | `generic_large`：1.20 × 1.00 m | 换机型时改 `config_1p5x4.json` 的 `robot_class` 重生成（6 档预设：go2 / a1 / b1 / anymal_c / g1 / generic_large） |
| 唤醒半径 | 0.6 m（≈840 颗/ env） | 按实测步频足迹调；见 §3 表，**别用 1.5 m** |

---

## 6. MuJoCo / mjlab 一页说明（不建议做主路径）

`layouts.npz` 是引擎无关的，但 MuJoCo 没有等价的容器：

1. **逐个 body + mesh geom**：`<body><freejoint/><geom type="mesh"/>`。`scale` 必须
   烘进顶点（MuJoCo 的 mesh 不支持逐实例非均匀缩放）→ 32 个 shape_id × N 种缩放组合要
   预先生成；或者退化成 sphere/ellipsoid 近似（精度掉一个量级）。
2. **数量上限**：4467 个自由体在 MuJoCo CPU 上基本跑不动（mjlab/warp 才有一点希望）。
   建议只在 MuJoCo 做 ≥10 倍缩小的验证（几百颗），主训练留在 Isaac。
3. **高度场**：`static/heightfield_L*.npy` → `hfield` 是唯一直接的路径
   （`cell = 5 mm`、`shape = 300×800`、`origin = (−0.75, −2.0)`、float32、米）。

---

## 7. 已知取舍与坑

1. **只验了 layout 0**（16 套里）。其余 15 套的物理行为未测；要跨布局泛化请自行补测。
2. **USD / `data.root_pos_w` 回读是假的**（fabric 开启时 USD 永不更新；IsaacLab 的
   buffered 量不调 `update()` 不刷新）。**只用 `make_pebble_view()` 读位姿**。
   误判指纹：位姿恒定且 `vz` 恰好等于 `2·g·dt`。
3. **4 层卵石（v3）**：v1 单层走完走廊有 33.4% 露出硬底板（宽洞 22.2%），v2 两层降到 9.3%
   （宽洞 5.0%），v3 床厚 0.12 m 后是 **3.0%（宽洞 0.52%，只有 8 个 3–4 cm 脚窝）**，走廊里床面
   5 百分位仍在硬面之上 16 mm。要回旧行为：删 `pebbles.layers` 里的层并调 `coverage`。
   v3 自沉降（0–30 步）：93.8% 的石头有位移、均值 17.4 mm（4 层堆叠第一次压实的量级），
   都发生在机器人起步前（前 30 步单独记账，不计入位移统计）。
5. **机器状态**：这台机 CPU 是 `powersave` profile 且多人共享，绝对耗时会 ±20% 波动；
   Kit 启动时间在 3~120 s 之间抖，批量跑要给足超时。
6. **相对路径**：`loader` 的 bundle 路径请显式传参，不要依赖 cwd。
