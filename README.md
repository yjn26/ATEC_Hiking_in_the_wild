# ATEC InstinctLab

面向 Unitree G1 的 GPU 强化学习训练、跑酷任务和 sim2sim 部署工程。本项目以 Isaac Lab 为仿真训练框架，`instinctlab` 提供环境侧扩展；`Hiking_deploy` 提供 MuJoCo + ONNX Runtime 部署与 sim2sim 流程。

上游参考：
- Hiking-in-the-wild: https://github.com/project-instinct/InstinctLab
- Hiking_deploy: https://github.com/Haydenil/Hiking_deploy

`Hiking_deploy` 已包含在本仓库中；直接克隆本仓库即可获得训练、部署和 sim2sim 所需的全部项目内容。

## 项目结构

```text
InstinctLab-isaaclab2.3.2/
├── source/instinctlab/instinctlab/     # InstinctLab Python 包
│   └── tasks/parkour/                  # G1 跑酷任务、奖励、观测和配置
├── scripts/instinct_rl/                # Instinct-RL 训练脚本
├── Hiking_deploy/                      # 部署与 sim2sim
│   ├── instinct_onboard/               # ONNX 策略推理和 ROS2 节点
│   ├── sim2sim/                        # MuJoCo G1 bridge、场景和仿真工具
│   ├── setup_env.sh                    # 实机部署环境配置
│   └── sim2sim/env_sim.sh              # loopback-only sim2sim 环境配置
└── DOCS.md                             # InstinctLab 扩展组件文档
```

## 主要能力

- **Isaac Lab 环境扩展**：`InstinctRlEnv`、多奖励组管理、监控、运动参考、虚拟障碍物和带噪声的分组射线传感器。
- **G1 Parkour 任务**：基于 AMP 的目标跟踪跑酷任务，包含地形、深度观测、课程学习、奖励项和终止条件。
- **模型导出与部署**：训练策略可以导出 ONNX，并由 `instinct_onboard` 执行推理。
- **sim2sim**：使用 MuJoCo、ROS2 和 ONNX Runtime 在部署前验证完整控制链路。

## 环境要求

### 训练

- Isaac Lab 2.3.2 及其运行依赖。
- 已安装本项目的 `instinctlab` 包与 `instinct_rl`。
- G1 跑酷任务所需的 AMASS/重定向运动数据。请在 `source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py` 的 `AmassMotionCfg` 中配置数据路径和运动筛选文件。

### Sim2Sim / 部署

- GPU 训练环境及 Isaac Lab 运行依赖。
- sim2sim 所需的 MuJoCo、ROS2、Python 和 ONNX Runtime 环境；详见 `Hiking_deploy/sim2sim/README.md`。
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

## Sim2Sim

进入仓库内的部署目录，并按 `Hiking_deploy/sim2sim/README.md` 完成依赖和场景准备：

```bash
cd Hiking_deploy
```

启动每个 sim2sim 终端前均先执行：

```bash
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

`sim2sim/env_sim.sh` 会将 DDS 绑定到 loopback 接口，因此该流程只能与本机 MuJoCo bridge 通信，不能连接实体 G1。详细的依赖、场景、地形生成和键盘控制说明见 `Hiking_deploy/sim2sim/README.md`。

## 实机部署

在实体机器人所在的 Ubuntu/WSL 环境中，使用 `source Hiking_deploy/setup_env.sh` 配置 ROS2、CycloneDDS、Unitree 消息和 Python 虚拟环境。先确认 DDS 网卡及 ROS2 话题正常，再从 dry-run 流程开始验证。不要在未核对网卡和策略检查点的情况下运行带 `--nodryrun` 的实机控制命令。

## 文档与扩展

- `DOCS.md`：多奖励、运动参考、虚拟障碍物和传感器组件的设计与配置示例。
- `source/instinctlab/instinctlab/tasks/parkour/README.md`：Parkour 任务的精简运行指南。
- 新任务应遵循 Isaac Lab 外部项目工作流，并使用 `instinctlab.envs:InstinctRlEnv` 注册环境。
