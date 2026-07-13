# affine_hafi

**Structured Formation Intention Learning: Continuous Affine Manifolds with Safety-Guaranteed Distributed MPC**

Research project extending HAFI (师姐 CoRL 2026 投稿) to learn team-level formation intention on a 6D SE(2) affine manifold, with a feasibility-preserving projection layer to the MPC feasible domain.

Also hosts a parallel backup direction (方向 B2): Communication-Robust Decentralized Formation Intention.

---

## 与部署包 D:\rl_mpc_deploy\ 的关系

| 项目 | 用途 |
|------|------|
| `D:\rl_mpc_deploy\` | **只读运行时** — 实车 ROS2 部署、推理节点、感知过滤 |
| `D:\affine_hafi\` (**本项目**) | **研究主仓库** — Env、训练、baselines、消融、评估 |

**接口**: 训完权重通过 `scripts/deploy_weight_to_realrobot.py` 拷进部署包的 `models/` 目录。

---

## 目录

```
configs/                # YAML 配置
├── affine_hafi.yaml    # 方向 A 主方法
├── comm_robust.yaml    # 方向 B2 备选
├── baselines/          # HAFI/AFOR/STAF 复现配置
└── scenarios/          # 6 类训练场景 + 4 killer scenarios

envs/                   # Gym Env（从零搭建，基于 config.py 规格）
├── formation_env.py    # 主 Env 类
├── scenario_generators/# 场景生成器
├── obstacle_dynamics.py
└── rewards.py

policies/               # 策略网络
├── affine_policy.py    # ★ A: 6D affine action head + SE(2) 约束
├── projection_layer.py # ★ A: feasibility-preserving projection
├── comm_gnn_policy.py  # ★ B2: decentralized GNN + emergent comm
├── dual_head_policy.py # 从部署包拷来的起点 (HAFI baseline)
└── feature_extractor.py# 22→128 双流

mpc/                    # 从部署包拷贝，独立演进
└── solver.py           # 后续可能改造以支持 affine reference

control/                # 从部署包拷贝
├── leader_node.py      # 原 HAFI leader 逻辑，作为 baseline 起点
├── mpc_node.py
└── interfaces.py

baselines/              # 对比方法复现
├── afor/               # AFOR (Deng ICRA'25) 重实现
├── staf/               # STAF (Deng CoRL'25) 重实现
├── mappo/
└── mpc_only/

train/                  # PPO 训练主循环 + curriculum + failure replay
├── train.py
├── curriculum.py
├── failure_replay.py
└── callbacks.py

eval/                   # 评估
├── eval.py
├── metrics.py
└── plot_trajectory.py

scripts/                # 一次性脚本
├── pilot_killer_scenarios.py  # ★ M2 止损 gate
├── deploy_weight_to_realrobot.py
└── verify_mpc_offline.py

experiments/            # 实验记录（.gitignore）
notebooks/              # 分析 Jupyter
tests/                  # pytest 单元测试
docs/                   # 方法记录 + 决策历史
config.py               # 从部署包拷贝，作为超参起点 + Env 规格
```

---

## 快速开始

**推荐（跨平台 conda）**:
```bash
conda env create -f environment.yml
conda activate affine_hafi

# 验证 MPC solver
python scripts/verify_mpc_offline.py

# Wandb 登录
wandb login
```

**pip fallback**:
```bash
pip install -r requirements.txt
```

**多机部署详情**：见 [docs/env_setup.md](docs/env_setup.md)（Windows dev / Ubuntu train / Ubuntu vehicle 三端配置）

---

## 方法文档

- **方向 A**: [docs/method_A_affine.md](docs/method_A_affine.md) — 6D affine manifold + projection + emergent
- **方向 B2**: [docs/method_B_comm.md](docs/method_B_comm.md) — decentralized + emergent comm + DR
- **决策历史**: [docs/decisions.md](docs/decisions.md)

## Milestone Timeline

| M | 内容 | 时间 | 止损 gate |
|---|---|---|---|
| M1 | 环境搭建 + baseline HAFI 复现 | 3-4 周 | HAFI SR 复现率 >90% |
| M2 | Affine action + projection + pilot | 4-6 周 | **Killer scenarios 上 vs HAFI gap >15%** |
| M3 | 完整训练 + AFOR/STAF baseline 复现 | 8-10 周 | AFOR/STAF 复现基本可信 |
| M4 | Ablation + Sim2Real + 视频 | 6-8 周 | 三个 demo 视频通过 |
| M5 | 论文写作 | 6-8 周 | 送审 |

---

## 关键决策

- **不依赖师姐训练仓库**（用户选择独立性优先）
- **MPC solver 拷贝独立演进**（不 import 部署包）
- **Env 从零搭**（基于 config.py 规格）
- **B2 并行小步推进**（M3 之前不占主要资源）

见 [docs/decisions.md](docs/decisions.md) 了解详情。
