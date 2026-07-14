"""
YAML config loader for training runs.

Usage:
    from train.config_loader import load_config
    cfg, extras = load_config("configs/affine_hafi.yaml")

`cfg` is a `config.Config` instance with the `env` block deep-merged into
the dataclass fields.

`extras` is a plain dict with the other top-level blocks:
    - method: {name, action_type, affine_theta_max, affine_kappa_max,
               enable_projection, projection_type, formation_entropy_weight}
    - ppo:    PPO hyperparameters (total_timesteps, n_envs, ...)
    - curriculum: {enabled, phase1_probs, phase2_probs, switch_step,
                   corridor_gap_start, corridor_gap_target, ...}
    - failure_replay: {enabled, max_prob, sr_low, sr_high, buffer_size}
    - logging: {wandb, wandb_project, wandb_entity, eval_freq, video_freq,
                checkpoint_freq}
    - env (optional): overrides for Config fields

The design keeps `Config` as the single source of truth for env-level
constants (matches deploy package), while training-loop configuration
(hyperparameters, callbacks) lives in `extras`.
"""

from __future__ import annotations
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from config import Config


def load_config(yaml_path: str) -> Tuple[Config, Dict[str, Any]]:
    """Load a YAML config and return (Config, extras_dict).

    Parameters
    ----------
    yaml_path : path to YAML file

    Returns
    -------
    cfg : Config with env overrides applied
    extras : dict containing method / ppo / curriculum / failure_replay / logging blocks
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"YAML must be a top-level dict, got {type(raw)}")

    # ============ Apply env overrides to Config ============
    cfg = Config()
    env_overrides = raw.get("env", {}) or {}
    if env_overrides:
        cfg = _apply_env_overrides(cfg, env_overrides)

    # ============ Extract extras ============
    extras: Dict[str, Any] = {
        "method": raw.get("method", {}) or {},
        "ppo": raw.get("ppo", {}) or {},
        "curriculum": raw.get("curriculum", {}) or {},
        "failure_replay": raw.get("failure_replay", {}) or {},
        "logging": raw.get("logging", {}) or {},
    }

    return cfg, extras


def _apply_env_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Apply dict overrides to Config dataclass, ignoring unknown fields."""
    valid_field_names = {f.name for f in fields(Config)}
    applied: Dict[str, Any] = {}
    for k, v in overrides.items():
        if k in valid_field_names:
            applied[k] = v
        # else: silently ignore (scenario_probs is not a Config field, it's a
        # separate dict used by FormationEnv init)
    return replace(cfg, **applied)


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    """Serialize Config to a plain dict (for wandb logging + snapshot)."""
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        # tuples → lists for YAML/JSON compat
        if isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


def snapshot_config(cfg: Config, extras: Dict[str, Any], out_dir: Path) -> None:
    """Write a config snapshot to `out_dir/config_snapshot.yaml`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = {
        "config": config_to_dict(cfg),
        **extras,
    }
    with (out_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(snap, f, allow_unicode=True, sort_keys=False)
