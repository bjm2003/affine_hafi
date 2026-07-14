# Ubuntu 训练机端启动流程

## 前提
- Ubuntu 20.04 或 22.04
- NVIDIA GPU（推荐）+ CUDA 12.1+ driver 已装
- SSH 到 GitHub 已配置（或 HTTPS + PAT）

## 首次部署（约 15 分钟）

### 1. Clone
```bash
cd ~
# 使用 SSH（推荐）
git clone git@github.com:bjm2003/affine_hafi.git
# 或 HTTPS
# git clone https://github.com/bjm2003/affine_hafi.git
cd affine_hafi
```

### 2. 装 conda（如果没装）
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc
```

### 3. 一键 setup（装 env + 验证 MPC + 跑 pytest）
```bash
bash scripts/setup_ubuntu.sh
```

**期望输出**：
- ✓ GPU detected: NVIDIA GeForce ... driver_version=...
- ✓ Env 'affine_hafi' created
- ✓ torch cuda_available=True
- ✓ MPC verify PASS
- ✓ 41 tests pass

**常见错误**：
- `conda not found` → 装 miniconda 后 `source ~/.bashrc`
- **`Solving environment` 时被 `已杀死 / Killed`（exit 137/OOM）** → conda 经典 SAT solver 吃内存被 OOM killer 干掉。`setup_ubuntu.sh` 现在会先自动装 `libmamba` solver（C++，内存占用低 10-100×）。若手动修复：
  ```bash
  conda install -n base -c conda-forge conda-libmamba-solver -y
  conda config --set solver libmamba
  bash scripts/setup_ubuntu.sh   # 重跑即可
  ```
- `MPC verify failed` → 一般是 CasADi 版本冲突，删 env 重装：`conda env remove -n affine_hafi && bash scripts/setup_ubuntu.sh`
- `pytest failed` → 报 error 贴给 Claude debug

### 4. Wandb 登录
```bash
wandb login
# 粘贴从 https://wandb.ai/authorize 拿的 API key
```

或环境变量方式：
```bash
export WANDB_API_KEY=<your_key>
echo "export WANDB_API_KEY=<your_key>" >> ~/.bashrc
```

## 启动 HAFI baseline 训练

```bash
bash scripts/launch_hafi_baseline.sh
```

**默认配置**：n_envs=12, device=cuda, seed=42, total=1.5M (from config)
**估算时间**：4060 或 3090/4090 单卡，约 6-15 小时（fps ~40-100）

**监控**（另开 SSH terminal）：
```bash
tail -f experiments/run_YYYYMMDD_HHMMSS_hafi_baseline/train_stdout.log
```
或直接看 wandb 面板：`https://wandb.ai/baijiaming46/affine_hafi`

**关注 sanity gates（前 200k 步内）**：
- `rollout/ep_rew_mean` 应从 -50 逐渐上升到 -10 以上
- `custom/mpc_feasibility_rate` > 0.95
- `custom/sr_per_scenario/open` 最先起来（易场景）
- 若 SanityHaltCallback 打 WARN → 检查 reward alignment（见 Phase D 剧本 docs/decisions.md）

**停止**：
```bash
kill $(cat experiments/run_XX/train.pid)
# 或强制 kill -9
```

## 训练完 → Gate G1 评估

```bash
python eval/eval.py --run experiments/run_YYYYMMDD_HHMMSS_hafi_baseline --tiers L1,L2,L3 --n_per_scenario 50
```

**期望结果**（对比 HAFI paper Table 1）：
| Tier | SR (ours) | SR (paper) | Gate G1 |
|------|-----------|------------|---------|
| L1 | ≥ 88.8% | 93.8% | ✓ pass |
| L2 | ≥ 85% | 91.8% | 参考 |
| L3 | ≥ 78% | 83.0% | 参考 |

若 L1 SR < 88.8% → 走 docs/decisions.md "Phase D debug playbook"。

## 训完后 → 把权重同步回 Windows 部署包（实车用）

```bash
# 在 Ubuntu 训练机
scp experiments/run_YYYYMMDD_HHMMSS_hafi_baseline/best_model/best_model.zip \
    windows_user@windows_ip:D:/rl_mpc_deploy/models/hafi_baseline_gate_g1.zip
```

或在 Windows 端：
```powershell
python D:/affine_hafi/scripts/deploy_weight_to_realrobot.py \
    --run experiments/run_YYYYMMDD_HHMMSS_hafi_baseline \
    --alias hafi_baseline_gate_g1
```

## 快速命令 cheat sheet

| 需求 | 命令 |
|------|------|
| 首次部署 | `bash scripts/setup_ubuntu.sh` |
| 启动训练 | `bash scripts/launch_hafi_baseline.sh` |
| 小规模验证（20 分钟）| `TOTAL=200000 bash scripts/launch_hafi_baseline.sh` |
| CPU-only 备胎 | `DEVICE=cpu N_ENVS=4 bash scripts/launch_hafi_baseline.sh` |
| 查训练日志 | `tail -f experiments/*/train_stdout.log` |
| 停止训练 | `kill $(cat experiments/*/train.pid)` |
| Gate G1 评估 | `python eval/eval.py --run experiments/... --tiers L1,L2,L3` |
| 从 Windows 端同步代码更新 | `git pull` （Ubuntu 端）|
