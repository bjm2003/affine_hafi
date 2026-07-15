"""KILLER SCENARIO 2 — Sequential Doorways

三道 0.6m 门框依次排列, 每道门方向逐个 90° 旋转。

预测: STAF 每次都要重新 subteam-split 又 merge, latency 累计;
      我们连续 shear+rotation 直接钻过去。

关键差异化: STAF 的 discrete graph cut 每次切换有开销;
           我们的 continuous affine manifold 无 switch latency。
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance, ObstacleStatic


class SequentialDoorwaysScenario(BaseScenario):
    name = "sequential_doorways"

    def __init__(
        self,
        n_doorways: int = 3,
        door_width: float = 0.6,
        door_spacing: float = 2.0,       # distance between successive doors
        wall_length: float = 3.0,        # extent of each wall on either side of door
    ):
        self.n_doorways = n_doorways
        self.door_width = door_width
        self.door_spacing = door_spacing
        self.wall_length = wall_length

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        n = self.n_doorways
        door_w = self.door_width
        spacing = self.door_spacing
        wall_len = self.wall_length

        wall_r = 0.12
        obs_spacing = 2 * wall_r + 0.05

        # Center each door on x-axis. Successive doors tilt progressively so the
        # formation must reorient/shear for each. Capped below 90°: a 90°-tilted
        # door on a straight x-axis path lays its wall ALONG the travel line and
        # would swallow the goal. 0/30/60° keeps every gap crossable and the goal
        # clear while still forcing reorientation (0.6m gap < 0.8m formation width).
        static_obs = []
        door_positions = []
        for i in range(n):
            angle = i * (np.pi / 6)   # 0, 30°, 60° for 3 doors
            door_center = np.array([i * spacing, 0.0])
            door_positions.append((door_center, angle))
            axis = np.array([np.cos(angle), np.sin(angle)])
            normal = np.array([-np.sin(angle), np.cos(angle)])

            n_wall = int(wall_len / obs_spacing / 2)
            for side_sign in (+1, -1):
                for k in range(n_wall):
                    lateral = (door_w / 2 + wall_r) + k * obs_spacing
                    pos = door_center + side_sign * lateral * normal
                    static_obs.append(ObstacleStatic(pos=pos, radius=wall_r))

        # Start before first door, goal past last door
        start_center = np.array([-1.5, 0.0])
        goal = np.array([(n - 1) * spacing + 1.5, 0.0])

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=0.0,
            static_obstacles=static_obs,
            dynamic_obstacles=[],
            metadata={
                "scenario": "sequential_doorways",
                "n_doorways": n,
                "door_width": door_w,
                "door_positions_and_angles": [(p.tolist(), float(a)) for p, a in door_positions],
            },
        )
