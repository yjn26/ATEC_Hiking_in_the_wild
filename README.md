# ATEC InstinctLab

面向 Unitree G1 的强化学习训练、跑酷任务和 sim2sim 部署工程。本项目以 Isaac Lab 为仿真训练框架，`instinctlab` 提供环境侧扩展；`Hiking_deploy` 提供在 Ubuntu 22.04 WSL2 上运行的 MuJoCo + ONNX Runtime CPU 部署与 sim2sim 流程。

上游参考：
- Hiking-in-the-wild: https://github.com/project-instinct/InstinctLab
- Hiking_deploy（CPU sim2sim 部署）是独立 Git 仓库，需要单独克隆：

```bash
git clone https://github.com/Haydenil/Hiking_deploy.git Hiking_deploy
```

注意：上游当前缺少两个 G1 elbow mesh 的 Git LFS 对象；在它们恢复前，完整的 CPU sim2sim 场景不可运行。

## 项目结构

```text
InstinctLab-isaaclab2.3.2/
├── source/instinctlab/instinctlab/     # InstinctLab Python 包
│   └── tasks/parkour/                  # G1 跑酷任务、奖励、观测和配置
├── scripts/instinct_rl/                # Instinct-RL 训练脚本
├── Hiking_deploy/                      # 独立 Git 仓库：CPU 部署与 sim2sim
│   ├── instinct_onboard/               # ONNX 策略推理和 ROS2 节点
│   ├── sim2sim/                        # MuJoCo G1 bridge、场景和仿真工具
│   ├── unitree_ros2/                   # Unitree ROS2 工作区
│   ├── bootstrap_env.sh                # WSL2 CPU 环境初始化
│   └── WSL2_CPU.md                     # CPU sim2sim 详细操作说明
└── DOCS.md                             # InstinctLab 扩展组件文档
```

## 主要能力

- **Isaac Lab 环境扩展**：`InstinctRlEnv`、多奖励组管理、监控、运动参考、虚拟障碍物和带噪声的分组射线传感器。
- **G1 Parkour 任务**：基于 AMP 的目标跟踪跑酷任务，包含地形、深度观测、课程学习、奖励项和终止条件。
- **模型导出与部署**：训练策略可以导出 ONNX，并由 `instinct_onboard` 在 CPU 上执行推理。
- **CPU sim2sim**：使用 MuJoCo、ROS2 Humble 和 ONNX Runtime，不依赖 Isaac Sim 或 NVIDIA GPU。

## 环境要求

### 训练

- Isaac Lab 2.3.2 及其运行依赖。
- 已安装本项目的 `instinctlab` 包与 `instinct_rl`。
- G1 跑酷任务所需的 AMASS/重定向运动数据。请在 `source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py` 的 `AmassMotionCfg` 中配置数据路径和运动筛选文件。

### CPU sim2sim / 部署

- Windows 上的 WSL2，Ubuntu 22.04。
- ROS2 Humble、Python 3.10；`Hiking_deploy/bootstrap_env.sh` 会安装或构建所需依赖。
- 首次部署前需检查 `Hiking_deploy/setup_env.sh` 中绑定到 G1 的网卡名称。

## G1 Parkour 训练

任务注册在 `source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py`：

| 任务 ID | 用途 |
| --- | --- |
| `Instinct-Parkour-Target-Amp-G1-v0` | 训练 G1 AMP 跑酷策略 |
| `Instinct-Parkour-Target-Amp-G1-Play-v0` | 回放和评估 |

在项目根目录执行训练：

```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0
```

回放已训练策略：

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-G1-v0 \
  --load_run=<run_name>
```

导出 ONNX：

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-G1-v0 \
  --load_run=<run_name> --exportonnx --useonnx
```

## WSL2 CPU sim2sim

进入部署仓库后初始化环境：

```bash
cd /mnt/d/Desktop_Files/InstinctLab-isaaclab2.3.2/Hiking_deploy
./bootstrap_env.sh
```

初始化完成后，打开三个 Ubuntu 22.04 WSL 终端；每个终端均先执行：

```bash
export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
cd /mnt/d/Desktop_Files/InstinctLab-isaaclab2.3.2/Hiking_deploy
source sim2sim/env_sim.sh
```

按以下顺序启动：

```bash
# 终端 1：MuJoCo G1 bridge 和 GUI
python sim2sim/g1_mujoco_bridge.py

# 终端 2：ONNX 策略
python instinct_onboard/scripts/g1_parkour_laptop.py \
  --zmq_addr tcp://127.0.0.1:5556 --nodryrun

# 终端 3（可选）：深度图和处理阶段查看器
python -m instinct_onboard.Tools.real_time_img \
  --source zmq --zmq-addr tcp://127.0.0.1:5556 --stages
```

`sim2sim/env_sim.sh` 会将 DDS 绑定到 loopback 接口，因此该流程只能与本机 MuJoCo bridge 通信，不能连接实体 G1。完整排障说明见 `Hiking_deploy/WSL2_CPU.md`。

## 实机部署

在实体机器人所在的 Ubuntu/WSL 环境中，使用 `source Hiking_deploy/setup_env.sh` 配置 ROS2、CycloneDDS、Unitree 消息和 Python 虚拟环境。先确认 DDS 网卡及 ROS2 话题正常，再从 dry-run 流程开始验证。不要在未核对网卡和策略检查点的情况下运行带 `--nodryrun` 的实机控制命令。

## 文档与扩展

- `DOCS.md`：多奖励、运动参考、虚拟障碍物和传感器组件的设计与配置示例。
- `source/instinctlab/instinctlab/tasks/parkour/README.md`：Parkour 任务的精简运行指南。
- 新任务应遵循 Isaac Lab 外部项目工作流，并使用 `instinctlab.envs:InstinctRlEnv` 注册环境。
