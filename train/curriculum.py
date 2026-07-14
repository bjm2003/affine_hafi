"""
Curriculum callback for FormationEnv training.

Two mechanisms:

1. **Scenario probability phases** (matches config.py:159-163)
   - Phase 1 (before switch_step): favor easier scenarios (open, u_trap, dynamic)
   - Phase 2 (after switch_step): balanced mix, more corridors

2. **Corridor gap annealing** (matches config.py:167-170)
   - Start at corridor_gap_start (e.g. 1.0 m)
   - Anneal to corridor_gap_target (e.g. 0.70 m)
   - Warmup: fraction of total_timesteps where gap stays at start
   - Anneal: fraction of total_timesteps over which gap linearly decreases

Callback ordering matters:
    [CurriculumCallback, FailureReplayCallback, ...]
Curriculum sets the BASE scenario_probs; failure replay boosts low-SR scenarios
ON TOP of that base.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.scenario_generators import TRAINING_SCENARIOS


class CurriculumCallback(BaseCallback):
    """Manage scenario probability phase switch + corridor gap annealing.

    Attributes updated on training envs (via set_attr):
        - scenario_probs: dict[str, float]
        - corridor_gap_override: float (or None)
    """

    def __init__(
        self,
        curriculum_cfg: Dict[str, Any],
        total_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.cfg = curriculum_cfg
        self.total_timesteps = int(total_timesteps)
        self.enabled = bool(curriculum_cfg.get("enabled", True))
        self._current_phase: Optional[int] = None    # 1 or 2
        self._current_gap: Optional[float] = None

    def _on_training_start(self) -> None:
        """Set initial phase 1 probs and initial gap."""
        if not self.enabled:
            return
        self._apply_phase(phase=1)
        gap0 = float(self.cfg.get("corridor_gap_start", 1.0))
        self._apply_gap(gap0)

    def _on_step(self) -> bool:
        if not self.enabled:
            return True

        t = self.num_timesteps
        total = self.total_timesteps

        # Phase switch
        switch_step = int(self.cfg.get("switch_step", 500_000))
        target_phase = 1 if t < switch_step else 2
        if target_phase != self._current_phase:
            self._apply_phase(target_phase)

        # Corridor gap annealing
        warmup_frac = float(self.cfg.get("corridor_gap_warmup_frac", 0.05))
        anneal_frac = float(self.cfg.get("corridor_gap_anneal_frac", 0.40))
        gap_start = float(self.cfg.get("corridor_gap_start", 1.0))
        gap_target = float(self.cfg.get("corridor_gap_target", 0.70))

        progress = t / max(total, 1)
        anneal_progress = float(np.clip(
            (progress - warmup_frac) / max(anneal_frac, 1e-9), 0.0, 1.0
        ))
        gap = gap_start + anneal_progress * (gap_target - gap_start)
        # Only push to envs if gap changed non-trivially
        if self._current_gap is None or abs(gap - self._current_gap) > 1e-3:
            self._apply_gap(gap)

        return True

    def _apply_phase(self, phase: int) -> None:
        """Push scenario_probs to all training envs."""
        key = "phase1_probs" if phase == 1 else "phase2_probs"
        probs_list = self.cfg.get(key)
        if probs_list is None:
            return
        assert len(probs_list) == len(TRAINING_SCENARIOS), (
            f"phase{phase}_probs length mismatch: {len(probs_list)} vs {len(TRAINING_SCENARIOS)}"
        )
        probs_dict = {name: float(p) for name, p in zip(TRAINING_SCENARIOS, probs_list)}
        self.training_env.set_attr("scenario_probs", probs_dict)
        self._current_phase = phase
        if self.verbose:
            print(f"[Curriculum] switched to phase {phase}: {probs_dict}")

    def _apply_gap(self, gap: float) -> None:
        """Push corridor_gap_override to all training envs."""
        self.training_env.set_attr("corridor_gap_override", float(gap))
        self._current_gap = float(gap)
        if self.verbose > 1:
            print(f"[Curriculum] corridor gap → {gap:.3f} m")

    def _on_rollout_end(self) -> None:
        """Log current curriculum state to SB3 logger (visible in wandb via sync)."""
        if not self.enabled:
            return
        self.logger.record("curriculum/phase", self._current_phase or 0)
        self.logger.record("curriculum/corridor_gap", self._current_gap or 0.0)
