# 多机环境部署指南

本项目部署在**三种机器**上，环境需求不同。用 conda `environment.yml` 统一管理，
个别机器按下表调整。

## 机器分工

| 机器 | 系统 | 用途 | GPU | 是否装训练环境 | 是否装 ROS2 |
|------|------|------|-----|--------------|------------|
| **Dev 开发机** | Windows 11 + Anaconda | 写代码 / debug / notebook 分析 | 4060 | ✅ 是（CUDA 版）| ❌ 否 |
| **Train 训练机** | Ubuntu 20.04 | PPO 训练 / 大规模评估 | 若有 GPU 装 GPU；无则 CPU | ✅ 是（CUDA/CPU 按硬件）| ❌ 否 |
| **小车 (Robot)** | Ubuntu 20.04 + ROS2 | 实车推理运行时 | 无 GPU | ⚠️ 只装推理必需（CPU）| ✅ 是（用部署包 `D:/rl_mpc_deploy/`） |

---

## Windows Dev 机（你目前的电脑）

已装 Anaconda。**推荐路线**：

```powershell
# 1. 创建 conda 环境（GPU 版，含 CUDA 12.1）
cd D:\affine_hafi
conda env create -f environment.yml
conda activate affine_hafi

# 2. 验证 PyTorch CUDA 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
# 期望输出: CUDA: True device: NVIDIA GeForce RTX 4060 ...

# 3. 验证 CasADi + IPOPT
python scripts/verify_mpc_offline.py
# 期望输出: MPC solve time < 50ms + [Verify] PASS

# 4. Wandb 登录
wandb login    # 粘贴 baijiaming46 账号的 API key
```

**Windows 特殊注意**:
- CasADi Windows JIT 需要 gcc（可选，不装也能跑，只是慢一些）。装法: 用 MSYS2 → `pacman -S mingw-w64-x86_64-gcc`，或者装 MinGW。
- 如果不想装 gcc，MPC solver 会自动 fallback 到解释模式（用户已见 `mpc/solver.py:54-59`）
- torch-geometric 在 Windows 首次装可能报错，通常 `pip install torch_geometric` 前先装 torch-scatter/sparse: `pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html`

---

## Ubuntu 20.04 训练机（另一台）

```bash
# 1. 装 miniconda (如果没装):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. 从 dev 机拷贝项目
scp -r D:/affine_hafi/ user@ubuntu-train:~/  # 或 rsync / git clone

# 3. 建环境
cd ~/affine_hafi
conda env create -f environment.yml
conda activate affine_hafi

# 4. 装 gcc (Ubuntu 天然有):
sudo apt install build-essential -y   # 如果 CasADi JIT 想加速

# 5. 验证
python scripts/verify_mpc_offline.py

# 6. Wandb 登录
wandb login
```

**Ubuntu 特殊注意**:
- 如果 Ubuntu 训练机是纯 CPU（无 GPU）：
  ```bash
  # 修改 environment.yml，把 pytorch-cuda=12.1 那行删掉，加：
  #   - cpuonly
  # 然后重建环境
  ```
- Ubuntu 20.04 系统 Python 是 3.8，conda 环境隔离后不影响

---

## 小车（ROS2 部署）

**小车不装研究项目**！只跑推理，用部署包 `D:/rl_mpc_deploy/`。

已有 README 详细说明。研究项目训练完的权重，通过：
```bash
# Dev 机上运行
python scripts/deploy_weight_to_realrobot.py --run experiments/run_XXX --alias affine_hafi_v1
# 结果: 权重拷到 D:/rl_mpc_deploy/models/affine_hafi_v1.zip

# 然后手动 scp / rsync 拷到小车
scp D:/rl_mpc_deploy/models/affine_hafi_v1.zip user@robot:~/rl_mpc_deploy/models/
```

**小车环境**（已由部署包管理，此处仅备忘）：
- Ubuntu 22.04 + ROS2 Humble
- Python 3.10+
- `pip install -r requirements.txt`（部署包里的，最小依赖）

---

## 关键版本锁定原因

| 包 | 版本 | 原因 |
|----|-----|------|
| Python | 3.10 | SB3 2.x 官方支持 3.8-3.11；Ubuntu 22.04 默认，兼容性最好 |
| numpy | <2.0 | SB3 / gymnasium 尚未完全兼容 numpy 2.x |
| PyTorch | 2.2-2.4 | 4060 需要 CUDA 12+；2.2+ 稳定；2.5 有 breaking changes |
| CUDA | 12.1 | 4060 原生支持；PyTorch 官方 cu121 wheel 齐全 |
| CasADi | ≥3.6 | HAFI 部署包用的版本，保证 MPC solver 权重兼容 |
| stable-baselines3 | 2.1-3.0 | 训练权重与部署包 `models/*.zip` 兼容 |

**SB3 版本至关重要**：训练机训好的 .zip 要能在小车上被同版 SB3 加载。约束：小车部署包 `requirements.txt` 说 `stable-baselines3>=2.1.0`，我们训练时锁 `<3.0` 以确保不引入 API 变更。

---

## 环境同步 checklist

每次 `environment.yml` 变动后：
```bash
# 三台机器都执行
conda env update -f environment.yml --prune
```

如果依赖问题严重，删了重建：
```bash
conda deactivate
conda env remove -n affine_hafi
conda env create -f environment.yml
```

---

## GPU 占用参考（4060 8GB / 16GB laptop 或桌面）

| 场景 | 显存需求 | 训练时间估算 |
|------|---------|------------|
| PPO with 12 envs + 22-obs MLP | ~2-3 GB | 1.5M steps ≈ 20-30h |
| Affine 6D + projection + 12 envs | ~3-4 GB | 3M steps ≈ 45-60h |
| Batch eval on 100 episodes × 4 scenarios | <1 GB | ~30 min |

**性能瓶颈通常在 CPU 侧**（环境模拟 + MPC IPOPT solve），GPU 只跑 policy forward。
4060 完全够用；训练慢主要因为 MPC IPOPT 单核 solve。

**建议**：如果训练太慢，看看 MPC solve time 分布，考虑：
- 减小 `mpc_horizon`（从 10 → 8）
- 关掉 IPOPT JIT（首次编译慢）
- 减小 `n_envs`（避免 CPU 打架）
- 用 vectorized fake env 加速 pilot 验证
