"""Straight corridor scenario:
   Two parallel walls forming a corridor of width `gap`, oriented along team motion direction.
   Team must contract to pass through.

   Wall = row of static obstacles (小圆代替长墙, 保持 n_obs_max=3 语义可管理)

   Gap width follows curriculum: cfg.corridor_gap_start → cfg.corridor_gap_target
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance, ObstacleStatic


class CorridorScenario(BaseScenario):
    name = "corridor"

    def __init__(self, gap_width: float = None):
        # If gap_width is None, sample from curriculum default range
        self.gap_width = gap_width

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        world_half = cfg.world_size / 2.0

        # Corridor gap (curriculum controlled)
        if self.gap_width is not None:
            gap = self.gap_width
        else:
            # Sample near target for training variety
            gap = float(rng.uniform(cfg.corridor_gap_target,
                                    cfg.corridor_gap_start))

        # Corridor axis: sample random orientation
        theta = float(rng.uniform(-np.pi, np.pi))
        axis = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])

        # Start and goal along corridor axis, both outside corridor
        dist_along = rng.uniform(*cfg.start_distance_range)
        start_center = -0.5 * dist_along * axis
        goal = 0.5 * dist_along * axis

        # Corridor length: 3-4m in the middle
        corridor_len = float(rng.uniform(2.5, 4.0))
        # Wall obstacles: two rows of static obs at ±gap/2 in normal direction
        wall_radius = float(rng.uniform(0.10, 0.20))
        spacing = 2 * wall_radius + 0.05
        n_per_side = int(corridor_len / spacing)

        static_obs = []
        for side_sign in (+1, -1):
            for k in range(n_per_side):
                along_offset = (k - n_per_side / 2 + 0.5) * spacing
                pos = along_offset * axis + side_sign * (gap / 2 + wall_radius) * normal
                static_obs.append(ObstacleStatic(pos=pos, radius=wall_radius))

        # Optionally 1 dynamic obstacle outside corridor
        n_dyn = int(rng.integers(0, 2))
        exclude = [(start_center, 1.5), (goal, 1.0)]
        # Exclude corridor tube
        for k in range(n_per_side + 2):
            along = (k - n_per_side / 2) * spacing
            exclude.append((along * axis, gap / 2 + 0.3))
        dynamic_obs = self.sample_dynamic_obstacles(
            rng, cfg, n=n_dyn, world_half=world_half, exclude_zones=exclude,
        )

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=theta,
            static_obstacles=static_obs,
            dynamic_obstacles=dynamic_obs,
            metadata={
                "scenario": "corridor",
                "gap_width": gap,
                "corridor_axis_theta": theta,
                "corridor_length": corridor_len,
            },
        )
