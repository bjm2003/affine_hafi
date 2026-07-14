"""
Custom metrics + wandb integration for training.

Provides:
    - init_wandb(): call once at train start
    - CustomMetricsCallback: rollup per-episode SR / MPC / reward-component metrics
    - SanityHaltCallback: warn if ep_rew_mean is anomalously low at midpoint
      (avoids wasting 25hr training on a reward-sign bug)

Depends on wandb (optional — training still runs without it).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def init_wandb(
    logging_cfg: Dict[str, Any],
    all_config: Dict[str, Any],
    exp_dir: str,
) -> Optional[Any]:
    """Initialize wandb if enabled + available. Returns wandb run handle or None."""
    if not _HAS_WANDB:
        print("[wandb] not installed; skipping")
        return None
    if not logging_cfg.get("wandb", False):
        return None

    project = logging_cfg.get("wandb_project", "affine_hafi")
    entity = logging_cfg.get("wandb_entity")
    run = wandb.init(
        project=project,
        entity=entity,
        config=all_config,
        dir=exp_dir,
        sync_tensorboard=True,
        save_code=True,
    )
    print(f"[wandb] run initialized: {run.name} (project={project}, entity={entity})")
    return run


class CustomMetricsCallback(BaseCallback):
    """Roll up per-episode & per-rollout custom metrics for wandb dashboard.

    Metrics emitted (via self.logger.record → syncs to wandb + tensorboard):
        - custom/ep_success_rate           mean across all completed episodes in rollout
        - custom/ep_collision_rate         mean across all completed episodes in rollout
        - custom/ep_mpc_feasibility_rate   mean per-step feasibility
        - custom/ep_mean_scale             mean scale across all steps
        - custom/reward_components/{name}  mean per-step contribution of each component
        - custom/sr_per_scenario/{name}    cumulative SR per scenario
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Rolling accumulators (reset each rollout)
        self._reset_rollup()
        # Per-scenario cumulative
        self._per_scenario_ep = {}
        self._per_scenario_succ = {}

    def _reset_rollup(self) -> None:
        self._n_ep = 0
        self._n_succ = 0
        self._n_coll = 0
        self._feasibility_sum = 0.0
        self._feasibility_n = 0
        self._scale_sum = 0.0
        self._scale_n = 0
        self._proj_gamma_sum = 0.0
        self._proj_active_n = 0
        self._proj_n = 0
        self._reward_comp_sum: Dict[str, float] = {}
        self._reward_comp_n: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            # Per-step accumulation
            if "mpc_feasibility_rate" in info:
                self._feasibility_sum += float(info["mpc_feasibility_rate"])
                self._feasibility_n += 1
            if "current_scale" in info:
                self._scale_sum += float(info["current_scale"])
                self._scale_n += 1
            if "projection_gamma" in info:
                self._proj_gamma_sum += float(info["projection_gamma"])
                self._proj_n += 1
                if bool(info.get("projection_active", False)):
                    self._proj_active_n += 1
            comp = info.get("reward_components", {})
            for k, v in comp.items():
                self._reward_comp_sum[k] = self._reward_comp_sum.get(k, 0.0) + float(v)
            self._reward_comp_n += 1

            # Episode termination
            if dones[i]:
                self._n_ep += 1
                success = bool(info.get("success", False))
                collision = bool(info.get("collision", False))
                if success:
                    self._n_succ += 1
                if collision:
                    self._n_coll += 1

                # Per-scenario
                scen = info.get("scenario_name", "unknown")
                self._per_scenario_ep[scen] = self._per_scenario_ep.get(scen, 0) + 1
                if success:
                    self._per_scenario_succ[scen] = self._per_scenario_succ.get(scen, 0) + 1

        return True

    def _on_rollout_end(self) -> None:
        # Aggregate rollout metrics
        if self._n_ep > 0:
            self.logger.record("custom/ep_success_rate", self._n_succ / self._n_ep)
            self.logger.record("custom/ep_collision_rate", self._n_coll / self._n_ep)

        if self._feasibility_n > 0:
            self.logger.record(
                "custom/mpc_feasibility_rate",
                self._feasibility_sum / self._feasibility_n,
            )
        if self._scale_n > 0:
            self.logger.record(
                "custom/mean_current_scale",
                self._scale_sum / self._scale_n,
            )
        if self._proj_n > 0:
            self.logger.record(
                "custom/projection_active_rate",
                self._proj_active_n / self._proj_n,
            )
            self.logger.record(
                "custom/mean_projection_gamma",
                self._proj_gamma_sum / self._proj_n,
            )
        if self._reward_comp_n > 0:
            for k, v in self._reward_comp_sum.items():
                self.logger.record(
                    f"custom/reward_components/{k}",
                    v / self._reward_comp_n,
                )

        # Per-scenario SR (cumulative across training so far)
        for scen, n in self._per_scenario_ep.items():
            if n > 0:
                sr = self._per_scenario_succ.get(scen, 0) / n
                self.logger.record(f"custom/sr_per_scenario/{scen}", sr)

        self._reset_rollup()


class SanityHaltCallback(BaseCallback):
    """Emit WARN if ep_rew_mean is anomalously low at a specified checkpoint step.

    Doesn't actually halt training — just prints a big warning so the user
    knows to check for a reward-sign / MPC-infeasibility bug before wasting
    25+ hours.
    """

    def __init__(
        self,
        check_at_step: int = 200_000,
        min_expected_ep_rew: float = -50.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_at = int(check_at_step)
        self.min_expected = float(min_expected_ep_rew)
        self._triggered = False

    def _on_step(self) -> bool:
        if self._triggered or self.num_timesteps < self.check_at:
            return True

        self._triggered = True

        # Try to read the SB3-tracked ep_rew_mean
        ep_rew = None
        for k, v in self.logger.name_to_value.items():
            if k == "rollout/ep_rew_mean":
                ep_rew = float(v)
                break

        if ep_rew is None:
            return True

        if ep_rew < self.min_expected:
            print("=" * 70)
            print(f"[SanityHalt] WARN: at step {self.num_timesteps}, "
                  f"ep_rew_mean={ep_rew:.2f} is below expected {self.min_expected:.2f}")
            print("[SanityHalt] Likely causes: reward sign bug, MPC infeasibility,")
            print("[SanityHalt] observation normalization mismatch, or corridor")
            print("[SanityHalt] scale weight too strong. Check wandb dashboard.")
            print("=" * 70)
        elif self.verbose:
            print(f"[SanityHalt] OK at step {self.num_timesteps}: ep_rew_mean={ep_rew:.2f}")

        return True
