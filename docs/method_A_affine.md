# Method A — Structured Formation Intention Learning: Affine Manifold

**Working title**: *Learning Affine Formation Intentions for Adaptive Multi-Robot Navigation*
**Target venue**: IROS 2027 / CoRL 2027（ICRA 2027 太紧）
**Status**: M1（环境搭建 + baseline 复现）

---

## 一句话定位

> HAFI 教编队"改变多大"；我们教编队"变成什么形状"—— 端到端从 6D 连续仿射流形上学习编队意图，并证明策略输出总能被投影到 MPC 可行域。

## Problem Statement

给定 N 台全向机器人和参考编队 $\mathcal{D}_0 = \{\mathbf{d}_i^0\}_{i=1}^N$，团队要在障碍环境中从起点抵达目标：
- **HAFI** 只学 1D scale $\sigma$ → 团队只能等比缩放
- **STAF** (CoRL 2025) 用 discrete graph cut 做 subteam split → 0/1 决策，不能连续过渡
- **AFOR** (ICRA 2025) 用 spring-damper 在 wedge/circle/line 三种预定义形状之间测试，但**不学它们之间的连续过渡**
- **Zhao 2018 / He&Jing 2025** 的仿射编队控制是 classical control，需要 hand-designed leader 参考

**我们的 problem**: 学一个 policy $\pi(z_t | s_t)$，输出 team-level 编队意图 $z_t \in \mathbb{R}^6$，参数化整个 team formation 在时刻 $t$ 的 SE(2) 仿射变换。

## Method (3 条贡献)

### C1 — Affine Formation Manifold 作为结构化动作空间

$$\mathcal{D}(z_t) = \mathbf{t}(z_t^{\text{trans}}) + R(\theta(z_t)) \cdot S(s_x(z_t), s_y(z_t)) \cdot K(\kappa(z_t)) \cdot \mathcal{D}_0$$

- $z_t = (\delta_x, \delta_y, \theta, s_x, s_y, \kappa) \in \mathbb{R}^6$
- 约束在 $\text{Aff}(2)$ 群上：policy 输出是李代数系数 $\mathfrak{aff}(2)$，指数映射回群
- 动作维度**不随 N 增长**（保留 HAFI 可扩展性）

**关键差异化**:
- vs STAF: continuous manifold vs discrete graph cut
- vs AFOR: 6D 联合学习 vs 1D scale + 预定义形状之间独立测试
- vs Zhao/He&Jing: 端到端学 vs hand-designed leader ref

### C2 — Feasibility-Preserving Intention Projection ★ 理论卖点

在 policy 输出和 MPC 输入之间插一个投影层：

$$\hat{z}_t = \arg\min_{z' \in \mathbb{R}^6} \|z' - z_t\|_M^2 \quad \text{s.t.} \quad \forall i, \mathcal{D}_i(z') \in \mathcal{F}_i(s_t)$$

其中 $\mathcal{F}_i$ 是车 $i$ 在 $s_t$ 下 MPC 的短时可行域。

**Proposition 1 (待证)**: 若 policy 输出 $z_t$ 使某车 nominal reference $\mathbf{p}_i^{\text{ref}} = \mathcal{D}_i(z_t) + \mathbf{p}_c^{\text{ref}}$ 落在 $\mathbf{p}_c^{\text{ref}}$ 附近的 free-space cluster 内，则投影 $\hat{z}_t$ 保证下层 MPC 有 non-slack 解，且 $\|\hat{z}_t - z_t\| \leq \epsilon(s_t)$。

**实现方式（pilot）**: 上式的一般形是逐障碍半平面约束的 QP。我们证明并实现其沿"尺寸轴"的 **1-D 特化 —— 最小各向同性收缩** $\gamma^\star = \max\{\gamma\in(0,1]: \forall i,\ \mathbf{p}_c^{\text{ref}} + \gamma\,\mathcal{D}_i(z_t)\in\mathcal{F}_i\}$，闭式、可微、保持策略选定的朝向/长宽比。
- numpy 精确版 `project_affine_offsets_np`（真值障碍几何）在 env rollout 时强制可行性；
- torch 可微版 `FeasibilityProjectionLayer`（垂直路径自由半宽约束）供 C2 消融与 differentiable-MPC。
- 完整 cvxpylayers QP 作为严格泛化留作对比/附录（见 `docs/decisions.md` 决策 8）。

**风险**: $\mathcal{F}_i$ 显式近似是难点，可从 velocity-space CBF 或 free-space clustering 出发；如果 tight version 证不下来，退到"probabilistic feasibility with bounded slack"。

### C3 — Emergent Structured Behaviors from Unified Reward

- **不加 mode switch**，不告诉 policy "corridor → line formation"
- **不加 shape-specific reward**
- 用 HAFI 一样的 reward（progress + goal + collision + formation + time），因为动作空间从 3D 变 6D，policy **涌现出**：
  - 直路 → keep aniso ≈ 1, shear ≈ 0
  - 窄廊 → $s_y \downarrow$（横向压扁），$\theta$ 跟 corridor 方向对齐
  - 弯道 → $\theta$ 动态旋转 + $\kappa$ 小幅 shear
  - 门框 → 局部 shear 让 leader 先钻

**回退方案**: 如果学不出结构化行为，加 formation entropy regularization $\mathcal{L}_{\text{ent}} = -\log\det(\mathcal{D}(z)^\top \mathcal{D}(z) + \epsilon I)$ 防塌陷。

## 4 个 Killer Scenarios (拉开与 AFOR/STAF gap)

### Scenario 1 — 弯曲窄廊 (Curved Slot Corridor)
- 走廊宽度 = 编队最紧宽度 + 5cm，弯道半径 1.5m
- 预测: HAFI 30-50% / AFOR 40-60% / STAF 50-65% / **Ours 80-90%**
- 论据: AFOR 无 rotation；STAF 过弯后重组失败

### Scenario 2 — 门框序列 (Sequential Doorways)
- 连续 3 道 0.6m 门框，方向逐个 90° 旋转
- 预测: 我们连续 shear+rotation 通过，STAF 每次 split-merge latency 累计

### Scenario 3 — 侧向不对称障碍密度 (Asymmetric Density Field)
- 一侧密一侧疏
- 论据: 均匀 scale 无法利用一侧空旷；我们 aniso-scale + $\delta_y$ 偏移靠向空旷侧
- 这是 He & Jing 2025 的经典 motivation 场景，但没人做 learning 版本

### Scenario 4 — 中期障碍注入 (Sudden Interior Injection)
- HAFI "I" 类障碍的加强版：注入在编队几何中心
- 论据: HAFI 只能 $\sigma$ 缩最小仍撞；STAF split 触发不及；**Ours** shear+rotation 瞬间避开中心

## 主实验矩阵

| Method | L1 | L2 | L3 | Curved Slot | Doorways | Asymmetric | Interior Inject |
|---|---|---|---|---|---|---|---|
| MPC (fixed) | 55 | 51 | 31 | ~15 | ~20 | ~35 | ~25 |
| IAPF | 77 | 73 | 51 | ~30 | ~40 | ~50 | ~40 |
| MAPPO | 81 | 67 | 56 | ~35 | ~30 | ~55 | ~30 |
| HAFI | 94 | 92 | 83 | ~50 | ~55 | ~70 | ~60 |
| AFOR | ~90 | ~85 | ~78 | ~50 | ~55 | ~72 | ~55 |
| STAF | ~92 | ~87 | ~80 | ~65 | ~60 | ~70 | ~50 |
| **Ours** | ~93 | ~91 | ~85 | **~85** | **~85** | **~90** | **~80** |

*预测数字，pilot 阶段必须验证。如果 4 个新场景上 gap 不足 20%，方法有硬伤要改。*

## Ablations

| Ablation | 意图 | 预期在哪些场景掉 |
|---|---|---|
| w/o shear (5D) | shear 真的有用？ | 门框、弯廊掉 5-10% |
| w/o rotation (5D) | 旋转必要吗？ | 弯廊、门框掉 15-20% |
| w/o aniso (4D iso-scale) | 就是 HAFI+rot？ | 侧向不对称掉 20%+ |
| unconstrained 6D | 结构化的价值？ | 训练慢 + 高 CR |
| w/o feasibility projection | C2 贡献？ | Sample efficiency 差 2-3× |
| discrete 5-template gating | 连续 vs 离散？ | 门框场景切换 latency 高 |
| w/o formation entropy reg | 会不会塌陷？ | 演示 topology collapse |

## Sim2Real Demos (用户手上硬件可做)

- **Demo A**: 三车三角形 → 遇门框 → **连续变形为一字纵队** → 出门 → 恢复三角形（视觉最强）
- **Demo B**: 三车过弯窄廊，编队 rotation 明显跟 corridor 方向对齐
- **Demo C**: 与真人动态障碍交互，aniso-scale 侧向压扁避人

## Timeline & 止损 Gate

| M | 内容 | 时间 | Gate |
|---|---|---|---|
| M1 | 环境搭建 + HAFI baseline 复现 | 3-4 周 | HAFI SR 复现 >90% |
| M2 | Affine action + projection + pilot | 4-6 周 | **Killer scenarios 上 gap >15%** |
| M3 | 完整训练 + AFOR/STAF baseline 复现 | 8-10 周 | AFOR/STAF 复现可信 |
| M4 | Ablation + Sim2Real + 视频 | 6-8 周 | 三视频过导师 review |
| M5 | 论文写作 | 6-8 周 | 送审 |

## 主要风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| AFOR/STAF 代码不开源 | 中 | 高 | 早期 email Hao Zhang 组；准备 re-implementation 备胎；实在不行引 paper 数字直接对比 |
| 6D 学不出 emergent 结构 | 中 | 中 | Formation entropy reg + curriculum（3D→6D 渐进放开）|
| Projection layer 求解慢 | 低 | 中 | Differentiable QP (cvxpylayers) 或松弛为 penalty |
| 仿射后 formation offsets 频繁 infeasible | 中 | 高 | Shear/aniso 上下界 + 投影层保底 |
| Reviewer: STAF 已解决 formation adaptation | 高 | 高 | 论文首段拉 continuous vs discrete 对比；实验有 discrete gating baseline 压制 |

## References (调研 2026-07-13)

- HAFI (师姐, 投 CoRL 2026) — 本地 `D:/rl_mpc_deploy/Supplementary materials/`
- AFOR — arXiv 2404.01618 — https://hcrlab.gitlab.io/project/afor/
- STAF — arXiv 2509.16412 — https://hcrlab.gitlab.io/project/STAF
- ReDiG — OpenReview WatS7243Zl
- Non-uniform Scaling (He & Jing) — arXiv 2508.02289
- Zhao 2018 IEEE TAC — 仿射编队开山之作
