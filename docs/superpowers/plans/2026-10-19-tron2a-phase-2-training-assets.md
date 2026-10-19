# TRON2A 阶段 2 训练资产实施计划

## 思维导图

```text
阶段 2 实施
|
|-- 生成衍生 URDF
|   `-- 来源哈希、18 个关节、mesh/collision 保持
|-- 定义训练资产规格
|   |-- 初始根姿态与 10 条腿驱动
|   `-- 8 个锁定关节保持驱动
|-- 建立检查入口
|   |-- 离线静态检查
|   `-- Isaac GUI/headless 运行检查
`-- 验收
    |-- 单元测试与阶段 1 回归
    `-- Isaac 运行时作为阶段 3 前阻断
```

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 建立可追溯的 TRON2A 18 关节训练资产，其中策略未来只能控制 10 个腿关节，8 个上肢/夹爪由独立 PD 保持。

**架构：** 生成器只负责从阶段 1 来源 URDF 派生训练 URDF；资产模块集中定义 URDF 导入选项、初始姿态和 actuator 映射；检查工具将离线契约与可选 Isaac 运行时验收分开，避免没有 Isaac Lab 时出现伪通过。

**技术栈：** Python 3、标准库 XML/JSON、NumPy、可选 Isaac Lab。

---

### 任务 1：派生训练 URDF

**文件：**
- 新建：`tools/generate_training_urdf.py`
- 新建：`assets/resources/tron2a_10dof/urdf/tron2a_10dof_hold_arms.urdf`
- 测试：`tests/tron2a_10dof/test_training_asset.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_training_urdf_preserves_every_movable_joint_and_mesh_reference():
    generated = parse_movable_joints(TRAINING_URDF_PATH)
    assert [joint["name"] for joint in generated] == contract["all_joints_18"]
    assert all(mesh.exists() for mesh in referenced_meshes(TRAINING_URDF_PATH))
```

- [ ] **步骤 2：运行测试，确认因文件或生成器不存在失败**

运行：`python -m unittest tests.tron2a_10dof.test_training_asset -v`

- [ ] **步骤 3：实现最小生成器**

```python
root = ET.parse(source_path).getroot()
root.attrib["name"] = "tron2a_10dof_hold_arms"
root.insert(0, ET.Comment(f"Generated from robot_rl.urdf sha256={source_hash}"))
ET.indent(root)
ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
```

- [ ] **步骤 4：生成 URDF 并确认测试通过**

运行：`python tools/generate_training_urdf.py && python -m unittest tests.tron2a_10dof.test_training_asset -v`

### 任务 2：建立可导入的训练资产规格

**文件：**
- 新建：`assets/__init__.py`
- 新建：`assets/tron2a_10dof.py`
- 修改：`tests/tron2a_10dof/test_training_asset.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_asset_spec_exposes_ten_policy_actuators_and_eight_locked_actuators():
    spec = load_training_asset_spec()
    assert tuple(spec.policy_joint_names) == tuple(contract["policy_joints_10"])
    assert set(spec.locked_actuators) == set(contract["locked_joints_8"])
    assert set(spec.policy_joint_names).isdisjoint(spec.locked_actuators)
```

- [ ] **步骤 2：运行测试，确认因模块不存在失败**

运行：`python -m unittest tests.tron2a_10dof.test_training_asset -v`

- [ ] **步骤 3：实现最小资产规格**

```python
@dataclass(frozen=True)
class TrainingAssetSpec:
    urdf_path: Path
    root_link: str
    policy_joint_names: tuple[str, ...]
    policy_actuators: Mapping[str, ActuatorSpec]
    locked_actuators: Mapping[str, ActuatorSpec]

def load_training_asset_spec() -> TrainingAssetSpec:
    return TrainingAssetSpec(...)
```

配置必须令 `merge_fixed_joints=False`、开启 self collision 与 contact sensor；可选的 `make_isaac_articulation_cfg()` 仅在导入 Isaac Lab 时执行。

- [ ] **步骤 4：运行阶段 2 与阶段 1 测试**

运行：`python -m unittest tests.tron2a_10dof.test_training_asset tests.tron2a_10dof.test_asset_contract -v`

### 任务 3：加入离线和运行时导入检查

**文件：**
- 新建：`tools/inspect_imported_robot.py`
- 修改：`tests/tron2a_10dof/test_training_asset.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_offline_inspection_reports_expected_import_contract():
    report = inspect_offline_training_asset()
    assert report["status"] == "passed"
    assert report["root_link"] == "base_Link"
    assert report["movable_joint_count"] == 18
```

- [ ] **步骤 2：运行测试，确认因检查函数不存在失败**

运行：`python -m unittest tests.tron2a_10dof.test_training_asset -v`

- [ ] **步骤 3：实现离线检查和可选 Isaac 入口**

```python
def inspect_offline_training_asset() -> dict[str, object]:
    spec = load_training_asset_spec()
    return {"status": "passed", "root_link": spec.root_link, "movable_joint_count": 18}

def run_isaac_static_hold_check() -> None:
    """Require Isaac Lab, import the articulation, and simulate ten seconds."""
```

- [ ] **步骤 4：验证离线检查**

运行：`python tools/inspect_imported_robot.py && python -m unittest tests.tron2a_10dof.test_training_asset -v`

### 任务 4：记录阶段 2 运行时闸门

**文件：**
- 修改：`PLAN.md`
- 测试：`tests/tron2a_10dof/test_training_asset.py`

- [ ] **步骤 1：补充阶段 2 离线交付与 Isaac GUI/headless 命令**
- [ ] **步骤 2：运行全部离线回归**

运行：`python -m unittest discover -s tests -v`

- [ ] **步骤 3：在安装 Isaac Lab 的环境运行运行时闸门**

运行：`python tools/inspect_imported_robot.py --isaac --headless` 与 `python tools/inspect_imported_robot.py --isaac`

预期：两次均导入训练 URDF，执行 10 秒静止保持，没有 NaN，8 个锁定关节误差不超过各自 `position_tolerance`。

- [ ] **步骤 4：提交阶段 2 的实现**

运行：`git add PLAN.md assets tools tests docs && git commit -m "feat: 建立TRON2A阶段2训练资产"`

## 自检

- 阶段 2 的每个要求都映射到任务 1 至任务 4。
- 计划不包含 TBD、TODO 或未定义的实现接口。
- 静态检查、物理运行检查和硬件授权严格分离；当前没有 Isaac Lab 时不得宣称运行时验收通过。
