# TRON2A 10DoF 从 0 到 1 构建计划

## 思维导图

```text
TRON2A 10DoF 构建
|
|-- 阶段 1（本次）：冻结输入与关节契约
|   |-- 来源软链接与 SHA256 清单
|   |-- URDF 自动解析
|   |-- 18 = 10 个策略关节 + 8 个锁定关节
|   `-- NPZ 名称、维度、帧率与哈希验证
|
|-- 阶段 2：训练资产导入与锁定关节 PD
|-- 阶段 3：10DoF 平地站立环境
|-- 阶段 4：AMP 数据投影与回放
|-- 阶段 5：平地 AMP 训练
|-- 阶段 6：地形与感知
|-- 阶段 7：ONNX 一致性
`-- 阶段 8-9：MuJoCo sim2sim 与实机前适配
```

## 原则

- TRON2A 使用独立的 `tron2a_10dof` 命名空间；不修改或复用 G1 的机器人配置、关节索引、动作缩放或模型。
- 机器人保留全部 18 个可动关节。策略只输出 10 个腿部关节动作；8 个上肢和夹爪关节由独立 PD 在固定目标处保持。
- 所有关节映射均按名称完成，禁止按数组前 10 列截断运动数据。
- NPZ 四元数已是 `wxyz`，后续加载器不得再次交换分量。
- 未完成 sim2sim 和硬件通信安全评审前，不连接实体设备。

## 阶段 1：冻结输入与建立关节契约

### 目标

在不启动 Isaac Sim 的前提下，固定可追溯的 TRON2A 资产和运动数据输入，建立唯一的 18 / 10 / 8 关节顺序，并以自动化验证阻断不一致的输入。

### 交付物

1. `assets/resources/tron2a_10dof/`：指向权威 URDF、mesh、MuJoCo XML 的只读软链接。
2. `assets/resources/tron2a_10dof/MANIFEST.json`：来源绝对路径、软链接、SHA256 与用途。
3. `tasks/tron2a_10dof_parkour/data/joint_contract.json`：由 URDF 顺序约束的 18 个关节、10 个策略关节和 8 个锁定关节。
4. `tasks/tron2a_10dof_parkour/data/locked_joint_pose.yaml`：锁定目标、允许误差、PD 参数和安全确认状态。
5. `tools/validate_asset_contract.py`：解析 URDF、检查哈希、关节划分及 NPZ 契约，并输出 JSON 报告。
6. `tests/tron2a_10dof/test_asset_contract.py`：无需 Isaac Sim 的回归测试。

### 锁定姿态决策

首期目标采用已记录的 TRON2A 仿真初始姿态：六个臂关节为 `0 rad`，两个夹爪分别为最大行程 `+0.05 m`、`-0.05 m`。这不是从运动数据帧推断的姿态，且逐项位于 URDF 限位内。该配置仍是仿真暂定值，进入阶段 2 的物理仿真前必须获得机械安全确认，确认前不得用于硬件。

### 阶段 1 验收

执行：

```bash
python tools/validate_asset_contract.py
python -m unittest tests.tron2a_10dof.test_asset_contract
```

必须同时满足：URDF 恰有 18 个非 fixed 关节；`18 = 10 + 8`，三组无重复且无遗漏；NPZ 的 `joint_names` 与 18 关节顺序逐项一致；运动文件为 18,317 帧、18 列、50 Hz，SHA256 为 `ddae0bd5726411efaeb090bb5999293f1dca3b28fcb277b4bc0b36dac5f66024`。

## 后续闸门

| 阶段 | 进入条件 | 退出条件 |
| --- | --- | --- |
| 2 | 阶段 1 契约通过，锁定姿态经机械确认 | Isaac 导入无缺失 mesh、无 NaN，锁定关节可保持 |
| 3 | 阶段 2 通过 | 平地站立环境 action 为 10，reset/step 无 NaN |
| 4 | 阶段 3 通过 | 按名称投影 AMP 数据，关节界与四元数校验通过 |
| 5 | 阶段 4 通过 | 平地 AMP 回放及训练张量维度正确 |
| 6 | 阶段 5 通过 | terrain/depth 单独固定种子回放正确 |
| 7 | 阶段 6 通过 | PyTorch 和 ONNX 的 10 维动作一致 |
| 8-9 | 阶段 7 通过 | sim2sim 后再单独评审真实硬件接口 |
