"""
Deterministic evaluation runner: play episodes with a trained model and
collect metrics-ready records.

Usage:
    from eval.eval_runner import run_episode, run_batch
    record = run_episode(env, model, deterministic=True)
    records = run_batch(env, model, n_episodes=100, seed=42)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np


def run_episode(
    env,
    model,
    deterministic: bool = True,
    seed: Optional[int] = None,
    scenario_override: Optional[str] = None,
    scenario_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one episode, return a metric-ready record dict.

    Fields returned:
        - success, collision, truncated, episode_length
        - formation_error_hist (per-step avg pairwise error, m)
        - control_effort (sum |u| — placeholder, currently uses velocity magnitude)
        - scenario_name
        - mpc_feasibility_rate (avg per-step feasibility)
        - dist_to_goal_final
    """
    reset_kwargs = {}
    if seed is not None:
        reset_kwargs["seed"] = int(seed)
    options = {}
    if scenario_override is not None:
        options["scenario"] = scenario_override
    if scenario_kwargs is not None:
        options["scenario_kwargs"] = scenario_kwargs
    if options:
        reset_kwargs["options"] = options

    obs, info = env.reset(**reset_kwargs)

    formation_err_hist: List[float] = []
    feasibility_hist: List[float] = []
    control_effort = 0.0
    step_count = 0
    max_steps = max_steps or env.cfg.max_episode_steps

    success = False
    collision = False
    truncated = False
    scenario_name = info.get("scenario_name", "unknown")

    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, trunc, info = env.step(action)

        # Formation error: mean pairwise deviation from scale-adjusted nominal
        pos = env.positions
        scale = float(info.get("current_scale", 1.0))
        errs = []
        for i in range(env.cfg.n_vehicles):
            for j in range(i + 1, env.cfg.n_vehicles):
                actual = float(np.linalg.norm(pos[i] - pos[j]))
                target = scale * float(env.nominal_dists[i, j])
                errs.append(abs(actual - target))
        if errs:
            formation_err_hist.append(float(np.mean(errs)))

        feasibility_hist.append(float(info.get("mpc_feasibility_rate", 1.0)))

        # Control effort proxy: not stored per-step in current env API,
        # so we use per-step displacement * n_vehicles as a stand-in.
        # (Full impl would sum |u| from MPC substep, tracked in info.)
        control_effort += float(env.cfg.v_max) * env.cfg.dt_rl

        step_count += 1
        if terminated or trunc:
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            truncated = bool(trunc)
            break

    if step_count >= max_steps and not (success or collision):
        truncated = True

    return {
        "success": success,
        "collision": collision,
        "truncated": truncated,
        "episode_length": step_count,
        "formation_error_hist": formation_err_hist,
        "control_effort": control_effort,
        "scenario_name": scenario_name,
        "mpc_feasibility_rate": float(np.mean(feasibility_hist)) if feasibility_hist else 1.0,
        "dist_to_goal_final": float(info.get("dist_to_goal", float("nan"))),
    }


def run_batch(
    env,
    model,
    n_episodes: int,
    deterministic: bool = True,
    seed_base: int = 0,
    scenario_override: Optional[str] = None,
    scenario_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run n_episodes independent episodes with deterministic seeds.

    Each episode uses seed = seed_base + i to guarantee reproducibility.
    """
    records = []
    for i in range(n_episodes):
        r = run_episode(
            env, model,
            deterministic=deterministic,
            seed=seed_base + i,
            scenario_override=scenario_override,
            scenario_kwargs=scenario_kwargs,
        )
        records.append(r)
    return records
