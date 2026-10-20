# MuJoCo 侧：同一床卵石（sim2sim）

这里的三份东西是给 MuJoCo 用的，由 `../export_mujoco.py` + `../config_mujoco_basin.json` 直接从
同一个生成器导出，石头形状、层数规则、摩擦/密度都和 Isaac 侧一致：

```
pebble_field_v3.xml          1199 个 <body>（**必须** include 在 <worldbody> 里面）
pebble_field_v3_assets.xml   1199 个 <mesh>（**必须** include 在 <mujoco> 顶层）
meshes/pebble_s00..31.obj    32 个单位鹅卵石形状（scale = 实际半轴）
```

## 为什么是两个文件

MJCF 的 `<include>` 是把文件的子元素插进**父元素**，所以资产（要进 `<asset>`）和本体（要进
`<worldbody>`）不能放同一个 include 里。旧版 `pebble_field_seed_20261019.xml` 是一份完整的
`<mujoco>` 文档，被 include 进来其实不合法——所以那份卵石从来没真的加载过。

## 接到场景里（`Hiking_deploy/sim2sim/assets/scene/scene_29dof_terrain_selector.xml`）

```xml
<mujoco>
  <include file="g1.xml"/>
  <include file="generated/pebble_field_v3_assets.xml"/>   <!-- 资产 -->
  ...
  <worldbody>
    ...
    <include file="generated/pebble_field_v3.xml"/>        <!-- 本体 -->
  </worldbody>
</mujoco>
```

同时把注释掉的盆（`pebble_basin_floor` + 四面墙 `pebble_basin_*`）取消注释。盆内是
x ∈ [−6.92, −4.08]、y ∈ [4.08, 6.92]、地面顶面 z = 0、围墙顶 0.30 m；这份导出的 `--offset`
已经写好是 (−5.5, 5.5)。

注意 `--mesh-prefix generated/meshes`：MuJoCo 的 asset 路径按**主模型文件**的目录解析，
不是按 include 文件自己的目录。

## 实测（MuJoCo CPU，dt = 4 ms，本机 RTX 机器）

| 配置 | 每步 | 相对实时 |
|---|---|---|
| 1199 颗 / 椭球碰撞 / freejoint | 9.7–11.8 ms | 34–41% |
| 3192 颗 | 742 ms | 0.5% |

- 9 cm 半径的球从 0.45 m 落下停在 z ≈ 0.18（床面 ≈0.21）——床吸收了冲击，没传到地面。
- Isaac 侧是 18 847 颗 2–6 cm；MuJoCo 这个数量级跑不动（约 0.2 ms/颗/步），所以这里是
  ~1.2k 颗、9–17 cm。要更细/更多只能用 `--max-pebbles` 换时间。
- `--collision mesh` 只适合几百颗：几千个独立 mesh 碰撞体直接 `Could not allocate memory`。

## 重新生成

```bash
python ../export_mujoco.py --config ../config_mujoco_basin.json --out . \
    --name pebble_field_v3 --footprint 2.84 2.84 --offset -5.5 5.5 \
    --collision ellipsoid --joints free --mesh-prefix generated/meshes
python ../bench_mujoco.py pebble_field_v3.xml --dt=0.004 --iters=20 --solver=PGS
```
