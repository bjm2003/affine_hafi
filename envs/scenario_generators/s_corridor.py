"""S-corridor: two staggered narrow gaps forcing team to compress + shift laterally.

Layout:
    Start ────╮
              │  ← gap 1 (offset up by +offset)
              ├───────╮
                      │  ← gap 2 (offset down by -offset)
                      ╰──── Goal

Compared to straight corridor: 需要 lateral maneuver + compress, 是 HAFI 的"s_corridor"场景。
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance, ObstacleStatic


class SCorridorScenario(BaseScenario):
    name = "s_corridor"

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        world_half = cfg.world_size / 2.0

        gap = float(rng.uniform(cfg.corridor_gap_target, cfg.corridor_gap_start))
        offset = float(rng.uniform(1.0, 1.8))  # lateral shift between two gaps
        seg_len = 2.5                           # length of each corridor segment

        theta = float(rng.uniform(-np.pi, np.pi))
        axis = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])

        # Positions of two gap centers along axis
        gap1_center = -0.75 * seg_len * axis + 0.5 * offset * normal
        gap2_center = +0.75 * seg_len * axis - 0.5 * offset * normal

        # Start / goal well outside
        dist_along = rng.uniform(*cfg.start_distance_range)
        start_center = -0.5 * dist_along * axis + 0.5 * offset * normal
        goal = 0.5 * dist_along * axis - 0.5 * offset * normal

        wall_r = 0.15
        spacing = 2 * wall_r + 0.05
        n_per_wall = int(seg_len / spacing)

        static_obs = []
        for gap_center, seg_normal_shift in [(gap1_center, +offset), (gap2_center, -offset)]:
            for side_sign in (+1, -1):
                for k in range(n_per_wall):
                    along = (k - n_per_wall / 2 + 0.5) * spacing
                    pos = gap_center + along * axis \
                        + side_sign * (gap / 2 + wall_r) * normal
                    static_obs.append(ObstacleStatic(pos=pos, radius=wall_r))

        # Sparse dynamic
        n_dyn = int(rng.integers(0, 2))
        exclude = [(start_center, 1.5), (goal, 1.0), (gap1_center, 1.5), (gap2_center, 1.5)]
        dynamic_obs = self.sample_dynamic_obstacles(
            rng, cfg, n=n_dyn, world_half=world_half, exclude_zones=exclude,
        )

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=theta,
            static_obstacles=static_obs,
            dynamic_obstacles=dynamic_obs,
            metadata={"scenario": "s_corridor", "gap_width": gap, "offset": offset},
        )
