"""Open field scenario: sparse static + 1 dynamic obstacle in a 12x12m arena."""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance


class OpenScenario(BaseScenario):
    name = "open"

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        world_half = cfg.world_size / 2.0

        start_center, goal = self.sample_start_goal_pair(rng, cfg)

        # Exclusion zone around start and goal (don't spawn obstacles too close)
        exclude = [
            (start_center, 1.5),
            (goal, 0.8),
        ]

        # HAFI L1 setting: 3-4 static, 1 dynamic, 1 injected
        n_static = int(rng.integers(3, 5))
        static_obs = self.sample_static_obstacles_uniform(
            rng, cfg, n=n_static, world_half=world_half, exclude_zones=exclude,
        )

        n_dyn = 1
        dynamic_obs = self.sample_dynamic_obstacles(
            rng, cfg, n=n_dyn, world_half=world_half, exclude_zones=exclude,
        )

        # Face goal at start
        goal_dir = goal - start_center
        start_orientation = float(np.arctan2(goal_dir[1], goal_dir[0]))

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=start_orientation,
            static_obstacles=static_obs,
            dynamic_obstacles=dynamic_obs,
            metadata={"scenario": "open", "n_static": n_static, "n_dynamic": n_dyn},
        )
