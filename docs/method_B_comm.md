# Method B2 — Communication-Robust Decentralized Formation Intention

**Working title**: *Decentralized Formation Intention Learning with Robust Emergent Communication*
**Target venue**: IROS 2027 / ICRA 2027（**主推**）；CoRL 2027（次选）
**Status**: Backup track，M3 之前不占主要资源

---

## 一句话定位

> HAFI 靠 leader 独裁 + 完美通信；我们让每台车都在通信丢包/延迟下涌现出鲁棒的分布式编队意图共识。

## Problem Statement

HAFI 有一个隐蔽的强假设：**centralized high-level policy**
- Leader 车集中收所有 pose + 一个 LiDAR，广播给所有车
- 训练时 v2v_dropout / delay / noise 全 0（部署包 config.py 保留了参数但训练时未启用）
- 论文 N=5 时 SR 掉到 60%，可能是 centralized state space 处理不了

## Novelty Gap

**空白**: 没有任何 formation navigation 论文把 "**learned formation intention** + **communication robustness training**" 组合起来:
- Formation 侧的（HAFI/AFOR/STAF）都假设 perfect comm
- Comm 侧的（MADE-Net / DMCA / CommNet）都做 navigation/exploration，不是 formation intention learning

## Method (3 条贡献)

### C1 — Decentralized Formation Intention with Emergent Consensus
- 每台车学同一份 policy（parameter sharing）：$\pi(z_t^i | s_t^i, m_t^{\mathcal{N}_i})$
- GNN attention 聚合邻居 message
- 通过局部 consensus 收敛到 team-level formation intention

### C2 — Bandwidth-Bottleneck Learned Communication
- Message 维度 4-8 bit（vs HAFI 广播完整 pose）
- 通过 Information Bottleneck 正则学：$I(m; s) < B$

### C3 — Communication Domain Randomization
- 训练时: dropout 0-50%、delay 0-10 steps、Gaussian noise
- 演示: 训好后 real-world dropout/delay 下不掉性能

## Baselines
- HAFI (centralized)
- MAPPO, IPPO
- CommNet, TarMAC, DGN
- Zero-shot 无 comm 版本

## Ablations
- w/o comm → 退化到 IPPO
- Bandwidth sweep: 2 / 4 / 8 / 16 bit
- w/o dropout training → 演示 catastrophic drop
- w/o consensus layer

## Robustness Sweep
- Dropout: 0% - 50%
- Delay: 0 - 10 steps
- Packet loss / burst noise
- **杀手场景**: 训练完后 kill 一台车通信，看队伍能否维持

## 实车 Demo
- 物理拔一台车 wifi，其他车继续跑
- 与 HAFI baseline 对比（HAFI 会瘫）

## Timeline

| 阶段 | 时间 |
|---|---|
| HAFI 改造成 decentralized 架构 | 3-4 周 |
| GNN + attention message layer | 3-4 周 |
| 训练（decentralized 慢）| 6-8 周 |
| 通信 DR + 消融 | 4-6 周 |
| 实车（拔 wifi demo）+ 视频 | 4-6 周 |
| 写作 | 6-8 周 |
| **合计** | **7-9 个月** |

## 主要风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Reviewer: 跟 CommNet 差在哪 | 高 | 中 | 明确 "formation-specific message" + 与 HAFI 混合评测 |
| Decentralized 训练慢 | 中 | 中 | 起步阶段 N=3 快速迭代 |
| Emergent comm 卷得厉害 | 高 | 低 | 强调 formation intention 是特色，通信只是手段 |

## 优势 (为什么并行)
- **部署包已经有 UDP 层 + config 参数**，工程复用度极高
- 实车 demo 强（物理拔 wifi）
- Novelty 空白最清晰
- 与 A 技术栈不重叠，M2 阶段可以并行不冲突

## References

- MADE-Net (RA-L 2022) — decentralized exploration with dropouts
- DMCA (arXiv 2022) — dense navigation with attention comm
- SCALE-COMM (arXiv 2605.27532) — MARL comm latent alignment
- CommNet / TarMAC / DGN — 经典 emergent comm baseline
