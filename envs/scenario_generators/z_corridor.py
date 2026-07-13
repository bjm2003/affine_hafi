"""Z-corridor: 90-degree bend forcing team to rotate through corner.

Layout:
    Start ──╮
            │
            │  ← vertical segment
            │
            ╰────────── Goal   ← horizontal segment

This scenario benefits our Affine method (rotation自由度) — HAFI can only compress.
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance, ObstacleStatic


class ZCorridorScenario(BaseScenario):
    name = "z_corridor"

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        world_half = cfg.world_size / 2.0

        gap = float(rng.uniform(cfg.corridor_gap_target, cfg.corridor_gap_start))
        seg_len = float(rng.uniform(2.0, 3.0))

        # Base orientation (Z bends 90° from this)
        theta = float(rng.uniform(-np.pi, np.pi))
        axis1 = np.array([np.cos(theta), np.sin(theta)])
        normal1 = np.array([-np.sin(theta), np.cos(theta)])
        # 90° bend direction
        axis2 = normal1
        normal2 = -axis1

        wall_r = 0.15
        spacing = 2 * wall_r + 0.05

        # Corner center
        corner = np.zeros(2)

        # Segment 1: from start toward corner along axis1
        seg1_start = corner - seg_len * axis1
        # Segment 2: from corner toward goal along axis2
        seg2_end = corner + seg_len * axis2

        start_center = seg1_start - 0.5 * axis1
        goal = seg2_end + 0.5 * axis2

        # Build wall obstacles for each segment
        static_obs = []
        for seg_start, seg_axis, seg_normal in [
            (seg1_start, axis1, normal1),
            (corner, axis2, normal2),
        ]:
            n_seg = int(seg_len / spacing)
            for side_sign in (+1, -1):
                for k in range(n_seg):
                    along = k * spacing
                    pos = seg_start + along * seg_axis \
                        + side_sign * (gap / 2 + wall_r) * seg_normal
                    static_obs.append(ObstacleStatic(pos=pos, radius=wall_r))

        n_dyn = 1
        exclude = [(start_center, 1.5), (goal, 1.0), (corner, 1.2)]
        dynamic_obs = self.sample_dynamic_obstacles(
            rng, cfg, n=n_dyn, world_half=world_half, exclude_zones=exclude,
        )

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=theta,
            static_obstacles=static_obs,
            dynamic_obstacles=dynamic_obs,
            metadata={"scenario": "z_corridor", "gap_width": gap, "corner_at": corner.tolist()},
        )
