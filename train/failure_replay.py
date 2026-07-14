"""
Failure replay callback: adaptive scenario re-weighting based on per-scenario SR EMA.

Design:
    - Maintain per-scenario success-rate EMA (`sr_ema`)
    - On rollout end, if `sr_ema[s] < sr_low`, BOOST that scenario's sampling
      probability toward `max_prob`
    - If `sr_ema[s] > sr_high`, DECAY back toward the curriculum base
    - Renormalize and push to envs via `set_attr("scenario_probs", ...)`

Why this design (vs transition replay):
    - On-policy PPO cannot re-inject transitions (breaks IS ratios)
    - Re-weighting scenarios is equivalent to spending more training budget on
      currently-failing distributions — matches the intent of config.py's
      `failure_replay_max_prob=0.3` parameter

Ordering:
    Curriculum sets a BASE distribution → FailureReplay boosts on top.
    In train.py callback list, order is [Curriculum, FailureReplay, ...].
"""

from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.scenario_generators import TRAINING_SCENARIOS


class FailureReplayCallback(BaseCallback):
    """Boost sampling of low-SR scenarios; decay high-SR scenarios."""

    def __init__(
        self,
        failure_cfg: Dict[str, Any],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.cfg = failure_cfg
        self.enabled = bool(failure_cfg.get("enabled", True))

        # EMA parameters
        self.ema_alpha = float(failure_cfg.get("ema_alpha", 0.05))
        self.sr_low = float(failure_cfg.get("sr_low", 0.5))
        self.sr_high = float(failure_cfg.get("sr_high", 0.8))
        self.max_prob = float(failure_cfg.get("max_prob", 0.3))

        # Per-scenario SR EMA (initialized to 1.0 so we don't panic-boost on first
        # few episodes before any data comes in)
        self.sr_ema: Dict[str, float] = {name: 1.0 for name in TRAINING_SCENARIOS}
        # Episode counts per scenario (for logging)
        self.n_ep: Dict[str, int] = {name: 0 for name in TRAINING_SCENARIOS}
        self.n_succ: Dict[str, int] = {name: 0 for name in TRAINING_SCENARIOS}

    def _on_step(self) -> bool:
        if not self.enabled:
            return True

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            # SB3 Monitor wraps final episode info under "episode" key at done
            if not dones[i]:
                continue
            # info fields we set: "success", "collision", "scenario_name"
            scenario = info.get("scenario_name")
            if scenario is None or scenario not in self.sr_ema:
                continue
            success = 1.0 if info.get("success", False) else 0.0
            self.sr_ema[scenario] = (
                (1.0 - self.ema_alpha) * self.sr_ema[scenario]
                + self.ema_alpha * success
            )
            self.n_ep[scenario] += 1
            self.n_succ[scenario] += int(success)

        return True

    def _on_rollout_end(self) -> None:
        if not self.enabled:
            return

        # Read current base scenario_probs (set by curriculum)
        base_probs = self.training_env.get_attr("scenario_probs")[0]  # dict
        base = dict(base_probs)   # copy

        # Compute boosted probs
        boosted = {}
        for name in TRAINING_SCENARIOS:
            b = float(base.get(name, 0.0))
            sr = self.sr_ema[name]
            if sr < self.sr_low:
                # Linearly interpolate between base and max_prob, based on how far
                # below sr_low the scenario is.
                gap = (self.sr_low - sr) / max(self.sr_low, 1e-6)
                boost = b + (self.max_prob - b) * float(np.clip(gap, 0.0, 1.0))
                boosted[name] = float(np.clip(boost, 0.0, self.max_prob))
            elif sr > self.sr_high:
                # Slight decay toward zero — actually just keep base
                boosted[name] = b
            else:
                boosted[name] = b

        # Renormalize
        total = sum(boosted.values())
        if total <= 0:
            # Fallback: uniform
            boosted = {n: 1.0 / len(TRAINING_SCENARIOS) for n in TRAINING_SCENARIOS}
        else:
            boosted = {k: v / total for k, v in boosted.items()}

        self.training_env.set_attr("scenario_probs", boosted)

        # Log
        for name, sr in self.sr_ema.items():
            self.logger.record(f"failure_replay/sr_ema/{name}", sr)
            self.logger.record(f"failure_replay/prob/{name}", boosted[name])

        if self.verbose:
            worst = min(self.sr_ema.items(), key=lambda kv: kv[1])
            print(f"[FailureReplay] worst scenario: {worst[0]} sr_ema={worst[1]:.3f} → prob={boosted[worst[0]]:.3f}")
