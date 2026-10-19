# TRON2A 阶段 2 训练资产设计

## 思维导图

```text
阶段 2：训练资产导入
|
|-- 权威输入：阶段 1 的 robot_rl.urdf 与关节契约
|-- 衍生训练 URDF：保留 18 个可动关节、mesh 与碰撞体
|-- Python 资产配置
|   |-- 10 条腿：独立 PD actuator，未来策略唯一动作候选
|   `-- 8 条上肢/夹爪：独立高刚度保持 actuator，不暴露给 action manager
|-- 导入检查
|   |-- 离线：名称、限位、mesh、配置映射
|   `-- Isaac：GUI/headless、静止 10 秒、无 NaN
`-- 闸门：未完成 Isaac 运行验收，不进入阶段 3
```

## 目标

从阶段 1 已冻结的 `robot_rl.urdf` 产生一个专用训练 URDF，并提供独立的 Isaac Lab articulation 配置和导入检查入口。它不注册任务、不训练策略，也不连接任何硬件。

## 方案选择

选择 URDF 作为训练资产权威来源。供应商 `robot.usd` 只在未来运行时检查中作为比对对象，不作为本阶段训练配置的输入。这样关节顺序、限位、mesh 相对路径和阶段 1 哈希可被同一条契约链追溯。

## 组件与边界

- `tools/generate_training_urdf.py`：从只读来源 URDF 生成训练 URDF；仅替换机器人名称并写入来源哈希注释，不删改 fixed、revolute 或 prismatic joint，也不改变 mesh/collision 节点。
- `assets/tron2a_10dof.py`：读取阶段 1 JSON/YAML（JSON 是 YAML 1.2 子集）并构造不可变的训练资产规格；Isaac Lab 可用时显式生成 `UrdfFileCfg` 和 `ArticulationCfg`。
- `tools/inspect_imported_robot.py`：离线模式检查衍生 URDF 和资产规格；`--isaac` 模式留给安装 Isaac Lab 的环境执行运行时导入与 10 秒静止保持检查。
- `tests/tron2a_10dof/test_training_asset.py`：覆盖生成结果、mesh 引用、关节划分、真实限位/effort/velocity/armature，以及锁定关节不属于策略集合。

## 物理约束

10 个腿部 actuator 从 `robot_rl.urdf` 的 effort 和 velocity 限位读取；转子惯量由同名 MuJoCo `armature` 数据冻结到配置中。8 个锁定关节使用阶段 1 的目标位置和 `800/40` 刚度/阻尼。初始根高度为 `0.85 m`，腿部直立且左右 yaw 分别为 `-pi/+pi`。这些是仿真暂定参数，不构成硬件安全授权。

## 验收与阻断

离线验收必须证明无缺失 mesh、18 = 10 + 8 且驱动参数与资产一致。安装 Isaac Lab 后，必须分别以 GUI 和 headless 模式运行导入检查；10 秒静止仿真不能出现 NaN，锁定关节误差必须不超过阶段 1 容差。当前环境缺少 `isaaclab`，故运行时验收不能被声称已完成，也不能进入阶段 3。
