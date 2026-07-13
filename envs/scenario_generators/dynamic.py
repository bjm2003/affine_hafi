"""Dynamic-heavy scenario: lots of moving obstacles crossing the team's path.

Used to train dynamic obstacle avoidance and formation robustness.
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance


class DynamicScenario(BaseScenario):
    name = "dynamic"

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        world_half = cfg.world_size / 2.0

        start_center, goal = self.sample_start_goal_pair(rng, cfg)
        goal_dir = goal - start_center
        theta = float(np.arctan2(goal_dir[1], goal_dir[0]))

        exclude = [(start_center, 1.5), (goal, 0.8)]

        n_static = int(rng.integers(2, 4))
        static_obs = self.sample_static_obstacles_uniform(
            rng, cfg, n=n_static, world_half=world_half, exclude_zones=exclude,
        )

        # Heavier dynamic: 2-3
        n_dyn = int(rng.integers(2, 4))
        dynamic_obs = self.sample_dynamic_obstacles(
            rng, cfg, n=n_dyn, world_half=world_half, exclude_zones=exclude,
        )

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=theta,
            static_obstacles=static_obs,
            dynamic_obstacles=dynamic_obs,
            metadata={"scenario": "dynamic", "n_dynamic": n_dyn},
        )
