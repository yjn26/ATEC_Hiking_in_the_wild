# 鹅卵石步行槽 + 抬升平台 场景生成器

在**任意有 Python 的机器**上生成与仿真引擎无关的场景包（bundle），再在装有
Isaac Lab / mjlab 的机器上几秒重建场景。生成端不需要 GPU、不需要大内存。

依赖：`numpy`、`scipy`（凸包/KD 树/旋转）。画图另需 `matplotlib`。
**不需要** trimesh / isaacsim / mujoco。

---

## 1. 场景几何

```
        Z
        ↑                      ┌──────────────┐  ← 站立台 (0.8 × 0.8, 顶面 0.255)
   0.255│                 ┌────┤              │
   0.25 │                 │  平台│  (0.80 ≤ X ≤ 1.80)
        │                 │    └──────────────┘
   0.20 │  ┌──┐  rim     │                    │
        │  │  │           │                    │
   0.15 │  │  │ ~ 卵石顶  │                    │   ← 平台对最高石头净空 = 0.10 m
   0.075│  │  │ ~ 填充面  │                    │   ← 填充面之上有 2 层卵石（床厚 0.075 m）
   0.00 │──┴──┴───────────┴────────────────────┤  ← 槽底
        └──┬──────────────────┬───────────────→ X
         -0.75              0.75  0.80      1.80
```

| 部件 | 尺寸 / 位置 |
|---|---|
| 槽内腔 | X ∈ [−0.75, 0.75]，Y ∈ [−2.0, 2.0]（1.5 m × 4.0 m） |
| 槽深 | 0.20 m（槽底 z=0，槽沿 z=0.20） |
| 槽壁 | 厚 0.05 m，四周封闭 |
| 卵石层 | **2 层堆叠**：层 0（长轴 26–60 mm，覆盖率 0.88）坐在填充体顶面，层 1（20–44 mm，覆盖率 0.70）压在层 0 的顶面上；最高石顶 = 0.15，床厚 = 0.15 − `fill_z ≈ 0.075` = 0.075 m；11297 颗 / 1883 颗·m⁻² |
| 填充体 | 槽内不透明实体（z: 0 → `fill_z ≈ 0.075`），是「砂浆垫层」，防止机器人沉穿槽底；现在它埋在两层石头下面 |
| 平台 | X ∈ [0.80, 1.80]，顶面 z = 0.25，**比最高石头高 0.10 m** |
| 站立台 | 平台上 1.2 × 1.0 m 薄板，顶面 0.255，默认中心 (1.40, −1.45) |
| 机器人出生点 | (1.40, −1.45, 0.255)，朝向 +Y；需向 −X 跨下约 0.12 m 进入卵石区 |

### 与机器人型号无关的部分（重要）

**卵石层、槽体、台阶高度、高度场、材质全部与机器人无关**——一套 bundle 可以给
Go2 / ANYmal / B1 / 人形共用。只有两处与机器人尺寸挂钩：

| 参数 | 为什么 | 默认 |
|---|---|---|
| 站立台尺寸 | 要站得下（脚不出界） | 1.2 × 1.0 m（最大机型通吃） |
| 平台宽度 | 要能转身 | 1.20 m |

需要紧身版本时用 `robot_class` 预设（一行切换，卵石层不受影响）：

```json
"robot_class": "unitree_go2"   // unitree_a1 / unitree_b1 / anymal_c /
                               // unitree_g1 / generic_large(默认)
```

| 预设 | 站立台 (X×Y) | 平台宽 |
|---|---|---|
| `unitree_go2` | 0.90 × 0.60 | 0.85 |
| `unitree_a1` | 0.80 × 0.55 | 0.80 |
| `unitree_b1` | 0.90 × 0.60 | 0.90 |
| `anymal_c` | 1.10 × 0.70 | 1.00 |
| `unitree_g1`（人形） | 0.90 × 0.70 | 0.95 |
| `generic_large`（默认） | 1.20 × 1.00 | 1.20 |

卵石：**2 层**，长轴 26~60 mm（层 0，覆盖率 0.88）/ 20~44 mm（层 1，覆盖率 0.70），
轴比 0.6~0.9 / 0.5~0.8，倾斜 ≤20°（层 1 用 26°），密度 2600 kg/m³ → 单颗约 2 g ~ 150 g，
平均约 20 g；合计 11297 颗 / 1883 颗·m⁻²，床厚 0.075 m。

> 坐标：右手系，**Z 向上，单位 m，四元数 `(w, x, y, z)`**。
> 这三条写在 `manifest.json` 里，跨引擎迁移时务必核对（scipy 的 `as_quat()` 是 `(x,y,z,w)`，
> 本生成器已转换）。

---

## 2. 用法

```powershell
# 冒烟测试（0.2×0.4 m，2 布局，1 秒）
python gen_pebbles.py --config config_smoke.json --out out_smoke

# 正式场景（1.5×4 m，16 布局，约 20 秒）
python gen_pebbles.py --config config_1p5x4.json --out out_full

# 只看单布局（快速预览几何）
python gen_pebbles.py --config config_1p5x4.json --out out_preview --layouts 1

# 铺装质量校验（最近邻距离 / 空隙率 / 精确椭圆窗口图）
python check_packing.py out_full

# 读 bundle 自检
python consume_example.py out_full
```

---

## 3. 输出清单（实测：单布局 11.3 MB，16 布局约 28 MB）

| 文件 | 说明 | 体积 |
|---|---|---|
| `manifest.json` | 版本、种子、坐标约定、物理建议值、sha256 | 2 KB |
| `layout_config.json` | 实际生效配置 + 派生几何量 | 2 KB |
| `shapes/rock_pool.npz` | 32 个单位凸包（verts/faces/offsets/volume） | 100 KB |
| `shapes/rock_pool.obj` | 同上，外部工具查看用 | 200 KB |
| **`layouts.npz`** | 每布局 `shape_id / pos / quat(wxyz) / scale / mass / mu` | 0.4 MB/布局 |
| `static/stage_boxes.json` | 槽/墙/平台/站立台/**bulk_fill** 的 box 列表 → **碰撞首选** | 1.5 KB |
| `static/stage.obj` | 同上网格版（仅渲染） | 2 KB |
| `static/cobble_L00.obj` | 布局 0 的合并卵石表面网格（渲染用，默认只出 1 个布局） | 9 MB |
| `static/heightfield_L##.npy` | 每布局高度场 `[nx, ny]`，i→X，j→Y，5 mm 栅格 | 0.96 MB/布局 |
| `static/heightfield_meta.json` | 原点/栅格/轴向 | 0.3 KB |
| `preview.png` | 俯视 + Y=0 剖面 | 300 KB |

**体积构成**：16 布局里 15 MB 是高度场、9 MB 是渲染网格，真正的「布局数据」只有
3 MB。这就是为什么可以随便换机器训练。

---

## 4. 关键参数旋钮

| 想要 | 改哪里 |
|---|---|
| 平台再高/再低 | `platform.top_height`（现 0.25 = 卵石最高点 0.15 + 0.10） |
| 卵石整体大小 | `pebbles.size_range`（现 0.026~0.06 m） |
| 更密/更疏 | 每层一个 `pebbles.layers[i].coverage`（现 层0=0.88、层1=0.70） |
| 层数 | 增删 `pebbles.layers` 条目；删光则回到 v1 单层行为 |
| 更粗糙 | `pebbles.axis_ratio_*` 拉大、`tilt_deg` 加大、`noise_amp` 加大 |
| 摩擦方差 | `materials.mu_range`（现 0.40~0.70） |
| 布局数量 | `output.n_layouts`（每个约 0.9 s） |
| 高度场精度 | `output.heightfield_cell`（5 mm；2 mm 体积涨 6 倍） |
| 站立台位置 | `standing_pad.center`（`null` = 自动放在平台起始端） |

**「更密」的物理上限**：铺装判据是**椭圆支撑函数之和**（两椭圆相切），不是圆。用
`max(a,b)` 当半径会退化到 93% 圆堆积（物理上不可能，六方最密仅 90.7%）。
`relax_iters` 是换覆盖率的代价：v1 用 150 只能到 0.75；v2 把底座提到 **0.88**，需要
`relax_iters=600`，得到每层仅 4 对残余重叠 >2 mm、最大 3.0 mm（11297 颗里）。
层 k>0 的落位不靠松弛：把前几层顶面栅格化成包络，石头只落在自身足迹内的**包络最大值**
上（`envelope_support`），从不手动制造互穿；落在孔洞上的石头靠重力自己沉进去。
`check_packing.py` 会打印最近邻距离与 2 cm 网格空隙率，用来验收。

---

## 5. 接入仿真引擎的要点

### Isaac Lab（推荐主力）

1. **静态几何用 `stage_boxes.json` 建 `CuboidCfg`**，不要用 `stage.obj`——盒子碰撞
   又便宜又稳。注意 `bulk_fill` 这一块不能省，它挡住了被埋住的下层。
2. **大规模训练用 heightfield**：`from_terrain`/自定义 `TerrainImporter`，一组 env
   共享一份高度场，prim 数只有 1。
3. **动态卵石只给少数 env**（≤256）：`RigidObjectCollectionCfg`，每个 env 一份同
   布局、不同 `pos` 偏移；`num_envs × n_pebble` 就是 prim 数，这是硬上限。
4. 卵石质量**显式写死**（`layouts.npz` 的 `mass`），别让引擎按尺寸猜。
5. 物理：`dt=1 ms`、`contact_offset=1~2 mm`、`rest_offset≈0.5 mm`、`position_iterations=8`、
   `restitution≈0.05`（卵石）/`≈0.1`（槽体）。默认 `contactOffset=2 cm` 与卵石（2~6 cm）同量级，
   做定量研究前必须调小。

### mjlab / MuJoCo

1. box 直接写 `geom type="box"`（注意 MuJoCo 的 size 是**半长**，json 里是全长，÷2）。
2. 卵石每颗一个 `freejoint` body + 1 个 mesh geom；**MuJoCo 的 mesh 不支持逐实例
   非均匀缩放**，两种处理：把 `scale` 烘进顶点（生成时按 scale 造几何），或者退化成
   sphere 近似。
3. 大规模训练同样用 `hfield`，不要 4000 颗自由石头。

`consume_example.py` 里有 `PebbleBundle` 读取类和两套引擎的接入片段（含
Isaac Sim 原生 USD 兜底脚本）。

---

## 6. 已知取舍（后面要接仿真时先读这段）

1. **2 层卵石，不是 1 层**。v1 是单层：4467 颗直接铺在 `fill_z` 上（石底离硬面只有
   0.3 mm），离线高度场（1 cm 格，"顶部离填充面 <6 mm"记作露底）量到**全槽 26.3% 的格子能
   看见硬底板**，机器人只要把脚下十几颗推开就站稳在硬面上，很容易学成"把石头挤开走池底"。
   v2 改两层：底部 0.88 覆盖率铺实、上层 0.70 压缝，`fill_z` 从 0.107 降到 0.075，
   **床厚 0.075 m**，露底降到 **4.7%**。同一个 3000 迭代策略各走一遍（700 步，`v2`/`v1`）：
   走过走廊的露底 v2 4.0% → 9.3%（走 2.2 m），v1 24.8% → 33.4%（走 2.6 m）。
   代价：动态刚体 4467 → 11297；单 env 演示照样跑（700 步 3.5 s 仿真约 4 min 含编码），
   多 env 训练要按需减层（删 `pebbles.layers` 第二条即回到 v1 行为）。
2. **不做物理沉降**。卵石按「坐下/压在前一层包络上」解析放置（椭球竖直半高精确求解 +
   包络局部最大值），生成 ~3 s/布局。启动后仍有 0–30 步自沉降（11297 颗里 10 颗位移
   >100 mm，均值 1.4 mm），属正常。如果要做强沉降：单布局单独开一个场景跑 300~800 步再
   序列化，**不要**在训练端每次 reset 时重跑（跨 GPU 不确定且慢 1000 倍）。
3. **高度场是近似**：用无噪声椭球的上包络，真实几何以 `cobble_L00.obj` /
   `rock_pool` + `layouts` 为准。精度 5 mm 时肉眼与碰撞都够。
4. **平台在 +X 侧**，机器人出生后要跨下约 12 cm 进槽。如果要改成从 Y 端进入（平地
   → 槽），把 `platform.side` 改了之后还需要手动调 `standing_pad.center`。
5. **摩擦是逐颗随机的**（0.40~0.70），但 `mu_rubber` 由引擎侧材质决定（建议 0.8~1.0），
   不在这里。

---

## 7. 下一步可选（已预留位置）

- `--settle`：接 Isaac Sim 做一次性沉降（当前未实现，结构上只需替换 `z` 的解析赋值）。
- `PebbleBundle.subset(li, x=..., y=...)`：取出「脚下走廊」子集，做动态/静态混合。
- 按几何族留出（star/ellipse/naca 那套思路）：把 `shape_id` 按族分组即可。

---

## 8. 物理体检（D1 静态 / D2 动态）与**回读陷阱**

`isaac_bench.py` 在真实 Isaac Sim 里建同一套 4467 prim，测构建/步进耗时、床层沉降
稳定性和 60 kg 冲击块。结果归档在 `out_full/physics_hints.json`。

> 下面的数字是 **v1 单层包（4467 颗）** 体检结果。v2 双层包（11297 颗）未重跑
> `isaac_bench.py`，步进耗时应按刚体数近似放大（实际 play.py 700 步含渲染/编码未见
> 明显卡顿）。

### 8.1 实测数字（Isaac Sim 5.1 + IsaacLab 0.54.3 + RTX 5090，dt = 1 ms，layout 0）

| 指标 | 静态 | 动态 |
|---|---|---|
| 建 4467 prim | 2.04 s（0.46 ms/个） | 2.43 s（0.54 ms/个） |
| `sim.reset()`（含 cooking） | 0.38 s | 0.53 s |
| 物理步进 | 1.50 ms → 666 步/s | **2.40 ms → 416 步/s** |

动态比静态慢 **1.6×**（不是预估的 5~10×），所以「动态卵石只给少数 env」的门槛比想象低，
但 256 env × 1000 颗仍然是 25 万刚体，激活半径（1.5~2 m）依旧是必需品。

床层稳定性（2 s 预热 + 2 s 观测）：下沉最大 0.79 mm、均值 0.003 mm、水平漂移最大
1.91 mm、无 NaN ——**解析放置的床层在仿真里是自洽的，不需要额外沉降**。

60 kg 块（0.50×0.30×0.18）从石顶上方 3 cm 落下，4 s 后：下沉 15.0 mm，块底面停在
0.1377 m，**高于石顶 p95 约 15 mm、高于填充体顶面 30 mm，没有穿透到填充体**；
24 颗卵石被推开，最大位移 38.7 mm，无速度爆炸（>100 m/s）。

### 8.2 回读陷阱（踩过，务必避开）

在本机（以及大多数 Isaac Sim 5.x + 无头模式）下，只有**最后一条**读法是真的：

| 读法 | 结果 | 原因 |
|---|---|---|
| `UsdGeom.Xformable(p).ComputeLocalToWorldTransform()` | **陈旧** | fabric 开启时 PhysX 只写 fabric，不写 USD |
| 同上 + `/physics/updateToUsd=true` + `update_transformations(False,True,True)` + `use_fabric=False` | **仍然陈旧** | 本机实测回写链路不通 |
| `RigidObject.data.root_pos_w` | **陈旧** | 这是缓冲量，不调 `asset.update(dt)` 就不刷新 |
| `SimulationManager.get_physics_sim_view().create_rigid_body_view(pattern).get_transforms()` | **✅ 实时** | 直读 PhysX |

误判特征：读到的位姿恒定不变，且 `vz` 恰好等于 `2·g·dt`（例如 −0.1635 = 2×9.81/120）
——这是 reset 期内积了 2 步的快照，不是「物理不积分」。

```python
from isaacsim.core.simulation_manager import SimulationManager
sv  = SimulationManager.get_physics_sim_view()
rbv = sv.create_rigid_body_view("/World/envs/env0/Pebbles/r*")
pos = rbv.get_transforms()[:, 0:3]          # (N,3)；前 3 列是位置
# 把物体强行归位（做冲击试验时在 drop 窗口开始才放下落体）：
rbv.set_transforms(torch.tensor([[x,y,z, 0,0,0,1]]), indices=torch.tensor([0], dtype=torch.int32))
```

注意 `set_transforms` 的四元数是 **xyzw**（单位旋转 = `0,0,0,1`），且必须传 `indices`。
另：`subscribe_physics_step_events()` 的返回值**必须用变量持有**，否则被 GC，回调静默失效。

### 8.3 复跑

```bash
bash run_homie.sh     # homie_lab（干净 IsaacLab main + 官方 kit），幂等，产物存在即跳过
```
