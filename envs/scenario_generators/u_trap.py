"""U-trap (dead-end): three-sided box with only one narrow entry (facing away from goal).

Layout (bird's eye):
                   Goal
                    ↑
             ┌──────────────┐
             │              │
             │     Start    │
             │              │
             └────╯    ╰────┘
                    ↑
                 (opening away from goal)

Team must detour around the trap to reach goal — tests avoidance planning.
Reward-wise this is a hard scenario (progress often negative during detour).
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance, ObstacleStatic


class UTrapScenario(BaseScenario):
    name = "u_trap"

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        world_half = cfg.world_size / 2.0

        # Trap dimensions
        trap_width = float(rng.uniform(2.0, 3.0))
        trap_depth = float(rng.uniform(1.5, 2.5))
        wall_r = 0.15
        spacing = 2 * wall_r + 0.05

        # Random orientation (opening direction)
        theta = float(rng.uniform(-np.pi, np.pi))
        forward = np.array([np.cos(theta), np.sin(theta)])
        right = np.array([np.sin(theta), -np.cos(theta)])

        # Trap center
        trap_center = np.zeros(2)

        # Start inside trap (near back wall, opening faces -forward)
        start_center = trap_center - 0.3 * trap_depth * forward
        # Goal in +forward direction, well past the trap opening
        dist_along = float(rng.uniform(*cfg.start_distance_range))
        goal = trap_center + dist_along * forward

        # Build walls: back + left + right (opening on -forward side)
        static_obs = []
        # Back wall (perpendicular to forward, at + trap_depth/2 * forward)
        n_back = int(trap_width / spacing)
        for k in range(n_back):
            offset = (k - n_back / 2 + 0.5) * spacing
            pos = trap_center + 0.5 * trap_depth * forward + offset * right
            static_obs.append(ObstacleStatic(pos=pos, radius=wall_r))
        # Left / right walls (perpendicular to right, spanning trap_depth)
        n_side = int(trap_depth / spacing)
        for side_sign in (+1, -1):
            for k in range(n_side):
                along = (k - n_side / 2 + 0.5) * spacing
                pos = trap_center + along * forward + side_sign * 0.5 * trap_width * right
                static_obs.append(ObstacleStatic(pos=pos, radius=wall_r))

        # Extra static outside the trap between trap and goal
        exclude = [(start_center, 1.0), (goal, 0.8), (trap_center, max(trap_width, trap_depth))]
        extra_static = self.sample_static_obstacles_uniform(
            rng, cfg, n=int(rng.integers(1, 3)),
            world_half=world_half, exclude_zones=exclude,
        )
        static_obs.extend(extra_static)

        n_dyn = int(rng.integers(0, 2))
        dynamic_obs = self.sample_dynamic_obstacles(
            rng, cfg, n=n_dyn, world_half=world_half, exclude_zones=exclude,
        )

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=theta,   # facing forward (toward goal, but blocked)
            static_obstacles=static_obs,
            dynamic_obstacles=dynamic_obs,
            metadata={"scenario": "u_trap", "trap_width": trap_width, "trap_depth": trap_depth},
        )
