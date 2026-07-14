# 决策历史

## 2026-07-13 —— 项目初始化决策

### 决策 1: 新建 D:\affine_hafi\ 作为 research 主仓库
**Options 考虑过**:
- (a) 在 `D:\rl_mpc_deploy\` 上直接改
- (b) 新建独立项目 ← **选中**
- (c) Git submodule 引用

**Reason**:
- 部署包 README 明确说是"最小运行时子集"，`pack_deploy.sh` 会**上游同步覆盖**——研究代码有丢失风险
- 部署包语义是"运行时"（推理 + ROS 节点），研究项目需要 Env / train / baselines / experiments
- 独立项目可以用现代化结构（pyproject / pytest / wandb），部署包不适合

**Trade-off**: 需要拷贝一些初始文件（MPC solver / config / policy），双份维护但边界清晰。

### 决策 2: MPC solver 拷贝一份到新项目独立演进
**Options**:
- (a) Python import `D:\rl_mpc_deploy\mpc\`
- (b) Git submodule
- (c) 拷贝独立演进 ← **选中**

**Reason**:
- Affine 方向可能需要改造 MPC 支持 affine reference 或 feasibility projection
- 部署包不是 git repo，submodule 不适用
- 拷贝独立演进保证不污染部署包

**约束**: 拷贝时机固定在 2026-07-13，后续如果部署包 MPC 有 upstream 更新，需要手动 merge。

### 决策 3: Gym Env 从零搭建，不依赖师姐训练仓库
**Options**:
- (a) 联系师姐要训练仓库
- (b) 从零搭建（基于 config.py 规格）← **选中**
- (c) 混合（先自搭 pilot，再要）

**Reason**:
- 用户明确表示要独立性优先
- config.py 已经定义了完整规格（6 类场景、curriculum、failure replay、reward 权重）
- 独立性带来**创新自由度**（不受师姐 Env 实现细节约束）
- 论文可以说 "we reimplement from spec"，reviewer 信任度更高

**Trade-off**: 时间成本 +4-6 周（M1 阶段主要花在 Env 搭建）。

### 决策 4: A + B2 并行推进，A 为重心
**Options 讨论过**:
- A + B1 (Neural CBF) — 两方向都吃硬理论，风险叠加
- A + B2 (Comm-Robust) ← **选中**
- A + B3 (Yaw-aware) — B3 novelty 弱

**Reason**:
- A 和 B2 技术栈不重叠（A 改动作空间/policy 结构，B2 改通信层/训练分布）
- B2 部署包 UDP 层可复用，启动成本几乎为 0
- 风险互补: A 卡在 AFOR/STAF 复现时 B2 独立不受影响

**并行时序**:
- M1-M2: A 为绝对主力，B2 只做架构准备
- M3: 各自 pilot 出结果后评估收敛还是双线并进

### 决策 5: 目标会议 IROS 2027 / CoRL 2027
**排除**:
- CoRL 2026: 时间来不及
- ICRA 2027 (Sep 2026 ddl): 太紧，AFOR/STAF baseline 复现可能来不及

**首选**:
- IROS 2027（应用+demo 强）
- CoRL 2027（method novelty 够就冲）

---

## Gate 定义

### Gate G1 (M1 结束, ~4 周)
- **通过条件**: 
  - HAFI baseline 在自搭 Env 上 SR 复现率 > 90%（L1 场景 vs 论文 93.8%）
  - MPC solver 拷贝跑通，与部署包结果一致
- **失败处理**: 
  - Env 规格有误 → 回 config.py 核对
  - MPC 结果不一致 → 检查配置差异

### Gate G2 (M2 结束, ~10 周) ★ **止损 Gate**
- **通过条件**: Affine baseline 在 4 killer scenarios 上 vs HAFI SR gap ≥ 15%
- **失败处理**: 
  - 如果 gap < 5%: 方向 A 有硬伤，**切换到 B2 为主**
  - 如果 gap 5-15%: 分析原因（可能 emergent 不出结构），加 entropy reg 再试
  - 如果 gap ≥ 15%: 继续 M3 完整训练

### Gate G3 (M3 结束, ~20 周)
- **通过条件**: AFOR/STAF baseline 已复现（至少能跑，SR 差异 < 10%）
- **失败处理**: 用 paper 中报告数字对比，注明"non-reproducible baseline, cited"

---

## 待决问题
- [x] 用户 GPU 资源？→ **单机单卡 3090/4090**（M1-M2 完全够；M3 全训 3M steps 预计 2-3 天）
- [x] 是否 email Hao Zhang 组要 AFOR/STAF 代码？→ **M1 baseline 跑通后再联系**（下个月）
- [x] 用户 Git 平台？→ **GitHub**
- [x] Wandb vs Tensorboard？→ **Wandb**（免费个人级）
- [ ] Wandb entity 用户名？（configs 里留空，用户 wandb login 后自动填）
- [ ] 项目 license？（研究项目一般 MIT / Apache-2.0，未定）

## 2026-07-13 —— 补充决策

### 决策 6: 资源与工具链
- GPU: **单机 4060**（用户实际配置，非之前假设的 3090/4090）
- 实验追踪: Wandb 免费版（entity: `baijiaming46`）
- Git: GitHub（本地 init，remote 用户自建）
- Baseline 联系时机: M1 通过后（Gate G1 达成 → HAFI SR ≥ 90%）联系 Hao Zhang 组

**训练时间估算 (4060)**:
- HAFI baseline 复现 1.5M steps: 约 20-30 小时
- Affine method 完整训练 3M steps: 约 45-60 小时
- 4 个 killer scenarios × 3 methods × 50 trials 评估: 约 30 分钟

（性能瓶颈在 CPU 侧的 MPC IPOPT solve，不在 GPU）

### 决策 7: 多机环境隔离策略
用户实际有三种机器：
- **Windows Dev**（本机 + Anaconda + 4060）— 写代码 / debug
- **Ubuntu 20.04 训练机**（另一台）— PPO 训练主力
- **Ubuntu 小车**（多台）— 实车推理（用部署包 `D:/rl_mpc_deploy/`）

**方案**:
- 项目主环境用 conda `environment.yml`（跨平台一致）
- `requirements.txt` 保留作 pip fallback
- 小车不装研究项目，只装部署包，权重通过 `deploy_weight_to_realrobot.py` 拷贝
- 详细步骤: `docs/env_setup.md`

**版本锁定**:
- Python 3.10（SB3 兼容 + Ubuntu 22.04 默认）
- PyTorch 2.2-2.4 + CUDA 12.1（4060 原生支持）
- numpy < 2.0（SB3 兼容性）
- stable-baselines3 2.1-3.0（与部署包权重加载兼容）

---

## 2026-07-14 —— M2 决策

### 决策 8: 可行性投影用"最小各向同性收缩"，不用 cvxpylayers QP
**Options**:
- (a) 可微 QP（cvxpylayers，逐障碍半平面约束）— 原 M2 plan 写法
- (b) 最小各向同性收缩 γ（沿 subgoal 收缩，直到每车参考清空障碍）← **选中**
- (c) SDP 松弛 / 学习式投影

**Reason**:
- **SB3 PPO 结构限制**: 动作在 rollout 时采样（numpy detach），策略更新用 `evaluate_actions` 重算 log_prob，不重跑 env。采样后再做确定性投影会破坏 log_prob 一致性（类似 SAC tanh 需 Jacobian 修正），QP 投影塞进 PPO 既脆弱又慢（每 forward 解一次 QP）。
- **各向同性收缩是 QP 的 1-D 特化**（沿"尺寸轴"），有闭式解、可微、O(N·n_obs)，且直接产生 killer 场景要的"缩小编队钻缝"涌现行为。
- 保留可解释性：收缩保持策略选的朝向 / 长宽比，只缩尺寸。

**实现**（镜像 affine decoder 的 torch+numpy 双份模式）:
- `policies/affine_decode_np.py::project_affine_offsets_np` — numpy 精确版（有真值障碍几何），在 `FormationEnv._decode_action` 里 rollout 时强制可行性。
- `policies/projection_layer.py::FeasibilityProjectionLayer` — torch 可微版（垂直路径自由半宽 w_allow 约束），供 C2 消融 + 未来 differentiable-MPC。
- 消融开关 `enable_projection`（method block + Config），wandb 记 `custom/projection_active_rate` + `custom/mean_projection_gamma`。

**Trade-off**: 论文里 QP 作为"严格泛化 / 附录"，主线用收缩。若 reviewer 质疑 novelty 不够，再补 QP 变体做对比。

### 决策 9: Ubuntu conda 环境必须用 libmamba solver
**问题**: conda 23.9.0 经典 SAT solver 在 4 channel + 多版本 pin 下"Solving environment"时被 OOM killer 杀掉（`已杀死`）。
**方案**: `setup_ubuntu.sh` 先装 `conda-libmamba-solver` 并 `--set solver libmamba`（C++ 版，内存低 10-100×），env create 传 `--solver libmamba` 兜底。
**How to apply**: 任何 Ubuntu 首次部署都走更新后的 `setup_ubuntu.sh`；手动修复见 `docs/ubuntu_training.md` 常见错误段。
