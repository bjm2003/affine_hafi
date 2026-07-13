"""
Main PPO training entry point for Method A (Affine) and Method B2 (Comm).

Usage:
    python train/train.py --config configs/affine_hafi.yaml
    python train/train.py --config configs/comm_robust.yaml
    python train/train.py --config configs/baselines/hafi_original.yaml

TODO:
    - YAML config loader (or Hydra if we later upgrade)
    - Wire up FormationEnv + AffinePolicy (or DualHeadPolicy for HAFI baseline)
    - SubprocVecEnv with cfg.n_envs parallel envs
    - PPO with SB3 (align with HAFI hyperparams from config.py)
    - Callbacks: EvalCallback / CheckpointCallback / VideoRecorder / WandbCallback
    - Curriculum: implement in train/curriculum.py
    - Failure replay: implement in train/failure_replay.py
    - Save checkpoint + config snapshot to experiments/run_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations
import argparse
import sys
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_envs", type=int, default=12)
    p.add_argument("--total_timesteps", type=int, default=None,
                   help="Override config total_timesteps")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Train] Loading config: {args.config}")
    print("[TODO] Env construction, policy build, PPO.learn() — implement in M1")


if __name__ == "__main__":
    main()
