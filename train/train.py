"""
Main PPO training entry point for Method A (Affine) and HAFI baseline.

Usage:
    python train/train.py --config configs/baselines/hafi_original.yaml
    python train/train.py --config configs/affine_hafi.yaml
    python train/train.py --config configs/affine_hafi.yaml --total_timesteps 10000  # smoke run

The training run creates:
    experiments/run_YYYYMMDD_HHMMSS/
        config_snapshot.yaml    # exact cfg + extras used
        git_sha.txt             # for reproducibility
        checkpoints/            # SB3 checkpoints (every 100k)
        best_model/             # EvalCallback's best_by_reward
        tb/                     # tensorboard logs (wandb syncs from here)
        train.log               # stdout/stderr
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

# Ensure project root is on path (script may be run from any cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_envs", type=int, default=None,
                   help="Override ppo.n_envs from config")
    p.add_argument("--total_timesteps", type=int, default=None,
                   help="Override ppo.total_timesteps (useful for smoke runs)")
    p.add_argument("--device", type=str, default="cuda",
                   help="'cuda' or 'cpu' — PPO forward pass device")
    p.add_argument("--wandb_off", action="store_true", help="Disable wandb even if config says on")
    p.add_argument("--exp_name", type=str, default=None, help="Override auto timestamp exp dir")
    return p.parse_args()


def make_env_fn(
    cfg,
    action_type: str,
    scenario_probs: dict,
    entropy_weight: float,
    affine_theta_max: float,
    affine_kappa_max: float,
    enable_projection: bool,
    rank: int,
    base_seed: int,
) -> Callable:
    """Return a callable for SubprocVecEnv — must be pickleable."""
    def _init():
        from envs.formation_env import FormationEnv
        env = FormationEnv(
            cfg=cfg,
            action_type=action_type,
            scenario_mode="mixed",
            scenario_probs=scenario_probs,
            entropy_weight=entropy_weight,
            affine_theta_max=affine_theta_max,
            affine_kappa_max=affine_kappa_max,
            enable_projection=enable_projection,
            seed=base_seed + rank * 1000,
        )
        return env
    return _init


def main():
    args = parse_args()

    # ============ Load config ============
    from train.config_loader import load_config, snapshot_config, config_to_dict

    cfg, extras = load_config(args.config)

    # CLI overrides
    ppo_cfg = dict(extras.get("ppo", {}))
    n_envs = int(args.n_envs) if args.n_envs is not None else int(ppo_cfg.get("n_envs", 12))
    total_timesteps = int(
        args.total_timesteps if args.total_timesteps is not None
        else ppo_cfg.get("total_timesteps", cfg.total_timesteps)
    )

    method_cfg = extras.get("method", {})
    action_type = str(method_cfg.get("action_type", "hafi_3d"))
    entropy_weight = float(method_cfg.get("formation_entropy_weight", 0.0))
    affine_theta_max = float(method_cfg.get("affine_theta_max", np.pi / 2))
    affine_kappa_max = float(method_cfg.get("affine_kappa_max", 0.5))
    enable_projection = bool(method_cfg.get("enable_projection", True))

    # ============ Set up experiment directory ============
    stamp = args.exp_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    exp_dir = PROJECT_ROOT / "experiments" / stamp
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "tb").mkdir(parents=True, exist_ok=True)

    # Snapshot config + git sha
    snapshot_config(cfg, extras, exp_dir)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
        ).strip()
    except Exception:
        git_sha = "unknown"
    (exp_dir / "git_sha.txt").write_text(git_sha)

    print(f"[Train] Experiment dir: {exp_dir}")
    print(f"[Train] Config: {args.config}")
    print(f"[Train] Action type: {action_type}, n_envs: {n_envs}, total: {total_timesteps}")
    print(f"[Train] Git SHA: {git_sha}")

    # ============ Build training envs ============
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    # Initial scenario probs (curriculum will override on training start)
    scenario_order = ["open", "corridor", "s_corridor", "z_corridor", "u_trap", "dynamic"]
    train_scenario_probs = {
        name: float(p) for name, p in zip(scenario_order, cfg.train_scenario_probs)
    }

    # Eval env distribution: use the curriculum's mature-phase mix (phase2) so
    # best_model selection is scored on scenarios the policy actually trains on.
    # The training_env gets phase2 via the curriculum callback, but the eval_env
    # is never touched by it — without this it would keep the Config default,
    # which for the isotropic baseline includes 15% unsolvable u_trap and just
    # injects noise into best_model selection. Falls back to the training mix
    # when no curriculum phase2 is configured.
    curriculum_cfg = extras.get("curriculum", {})
    phase2_probs = curriculum_cfg.get("phase2_probs")
    if curriculum_cfg.get("enabled", False) and phase2_probs:
        eval_scenario_probs = {
            name: float(p) for name, p in zip(scenario_order, phase2_probs)
        }
    else:
        eval_scenario_probs = train_scenario_probs

    env_fns = [
        make_env_fn(
            cfg=cfg,
            action_type=action_type,
            scenario_probs=train_scenario_probs,
            entropy_weight=entropy_weight,
            affine_theta_max=affine_theta_max,
            affine_kappa_max=affine_kappa_max,
            enable_projection=enable_projection,
            rank=i,
            base_seed=args.seed,
        )
        for i in range(n_envs)
    ]

    if n_envs == 1:
        # DummyVecEnv is faster + simpler for smoke runs (no subprocess overhead)
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv(env_fns)
    else:
        train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env, filename=str(exp_dir / "monitor.csv"))

    # Eval env (single, always DummyVecEnv)
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([
        make_env_fn(
            cfg=cfg, action_type=action_type,
            scenario_probs=eval_scenario_probs,
            entropy_weight=entropy_weight,
            affine_theta_max=affine_theta_max,
            affine_kappa_max=affine_kappa_max,
            enable_projection=enable_projection,
            rank=0,
            base_seed=args.seed + 100_000,
        )
    ])
    eval_env = VecMonitor(eval_env)

    # ============ Build PPO ============
    from stable_baselines3 import PPO
    from policies.feature_extractor import FormationFeatureExtractor

    if action_type == "hafi_3d":
        from policies.dual_head_policy import DualHeadPolicy
        policy_cls = DualHeadPolicy
        policy_kwargs = {
            "features_extractor_class": FormationFeatureExtractor,
            "features_extractor_kwargs": {"cfg": cfg},
            "n_spatial_dirs": cfg.n_spatial_dirs,
        }
    elif action_type == "affine_6d":
        from policies.affine_policy import AffinePolicy
        policy_cls = AffinePolicy
        policy_kwargs = {
            "features_extractor_class": FormationFeatureExtractor,
            "features_extractor_kwargs": {"cfg": cfg},
            "affine_theta_max": affine_theta_max,
            "affine_kappa_max": affine_kappa_max,
        }
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

    model = PPO(
        policy=policy_cls,
        env=train_env,
        learning_rate=float(ppo_cfg.get("lr", cfg.lr)),
        n_steps=int(ppo_cfg.get("n_steps", cfg.n_steps)) // n_envs,   # SB3 semantics: per-env
        batch_size=int(ppo_cfg.get("batch_size", cfg.batch_size)),
        n_epochs=int(ppo_cfg.get("n_epochs", cfg.n_epochs)),
        gamma=float(ppo_cfg.get("gamma", cfg.gamma)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", cfg.gae_lambda)),
        clip_range=float(ppo_cfg.get("clip_range", cfg.clip_range)),
        ent_coef=float(ppo_cfg.get("ent_coef", cfg.ent_coef)),
        tensorboard_log=str(exp_dir / "tb"),
        policy_kwargs=policy_kwargs,
        device=args.device,
        verbose=1,
        seed=args.seed,
    )

    # ============ Callbacks ============
    from stable_baselines3.common.callbacks import (
        CallbackList, CheckpointCallback, EvalCallback,
    )
    from train.curriculum import CurriculumCallback
    from train.failure_replay import FailureReplayCallback
    from train.logging_hooks import (
        CustomMetricsCallback, SanityHaltCallback, init_wandb,
    )

    logging_cfg = extras.get("logging", {})
    if args.wandb_off:
        logging_cfg = dict(logging_cfg)
        logging_cfg["wandb"] = False

    checkpoint_freq_env_steps = int(logging_cfg.get("checkpoint_freq", 100_000))
    eval_freq_env_steps = int(logging_cfg.get("eval_freq", 50_000))
    # SB3 EvalCallback / CheckpointCallback expects per-env-worker frequency:
    checkpoint_freq_worker = max(1, checkpoint_freq_env_steps // n_envs)
    eval_freq_worker = max(1, eval_freq_env_steps // n_envs)

    callbacks = [
        CurriculumCallback(
            curriculum_cfg=extras.get("curriculum", {}),
            total_timesteps=total_timesteps,
            verbose=1,
        ),
        FailureReplayCallback(
            failure_cfg=extras.get("failure_replay", {}),
            verbose=1,
        ),
        CustomMetricsCallback(verbose=0),
        SanityHaltCallback(check_at_step=200_000, min_expected_ep_rew=-50.0, verbose=1),
        CheckpointCallback(
            save_freq=checkpoint_freq_worker,
            save_path=str(exp_dir / "checkpoints"),
            name_prefix="rl_model",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(exp_dir / "best_model"),
            log_path=str(exp_dir / "eval"),
            eval_freq=eval_freq_worker,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
    ]

    # ============ Init wandb (must happen AFTER PPO to sync TB) ============
    all_config = {
        "cfg": config_to_dict(cfg),
        **extras,
        "cli": {
            "seed": args.seed,
            "n_envs": n_envs,
            "total_timesteps": total_timesteps,
            "device": args.device,
        },
        "git_sha": git_sha,
    }
    wandb_run = init_wandb(logging_cfg, all_config, str(exp_dir))
    if wandb_run is not None:
        try:
            from wandb.integration.sb3 import WandbCallback
            callbacks.append(WandbCallback(
                model_save_path=str(exp_dir / "wandb_ckpts"),
                model_save_freq=checkpoint_freq_worker,
                verbose=1,
            ))
        except ImportError:
            print("[wandb] wandb.integration.sb3 not available; TB sync only")

    # ============ Train ============
    print(f"[Train] Starting learn(): {total_timesteps} timesteps, {n_envs} envs, "
          f"~{total_timesteps // n_envs} steps per worker")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            log_interval=1,
        )
    finally:
        # Always save the final model, even on Ctrl+C
        final_path = exp_dir / "final_model.zip"
        model.save(str(final_path))
        print(f"[Train] Saved final model to {final_path}")
        if wandb_run is not None:
            wandb_run.finish()

    print(f"[Train] Complete. Experiment dir: {exp_dir}")


if __name__ == "__main__":
    main()
