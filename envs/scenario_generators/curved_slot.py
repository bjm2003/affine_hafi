"""KILLER SCENARIO 1 — Curved Slot Corridor

弯曲窄廊。走廊沿着弧线弯曲, 宽度 = 编队最紧宽度 + 5cm, 弯道半径 1.5m。

预测结果:
    HAFI (isotropic scale): SR 30-50% (弯道时外侧车撞墙)
    AFOR (spring-damper):   SR 40-60% (无 rotation 自由度)
    STAF (subteam split):   SR 50-65% (过弯后重组失败率高)
    Ours (Affine):          SR 80-90% (aniso-scale 压扁 + rotation 跟弯)

关键差异化: AFOR 无 rotation; STAF 的 graph cut 只做拓扑分/合, 不做旋转对齐
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance, ObstacleStatic


class CurvedSlotScenario(BaseScenario):
    name = "curved_slot"

    def __init__(
        self,
        arc_radius: float = 1.5,
        arc_span: float = np.pi / 2,      # 90° bend
        gap_width: float = 0.7,           # tightest ≈ 2*(vehicle_radius + d_safe) + 5cm
    ):
        self.arc_radius = arc_radius
        self.arc_span = arc_span
        self.gap_width = gap_width

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        arc_r = self.arc_radius
        gap = self.gap_width
        span = self.arc_span

        # Arc center at origin. Corridor axis is tangent along the arc.
        # Discretize arc into wall obstacles.
        wall_r = 0.10
        arc_length = arc_r * span
        spacing = 2 * wall_r + 0.05
        n_along = max(int(arc_length / spacing), 5)

        static_obs = []
        # Approach segment (straight into arc)
        pre_length = 1.5
        n_pre = int(pre_length / spacing)
        for k in range(n_pre):
            along = -pre_length + k * spacing
            for side in (+1, -1):
                pos = np.array([
                    along,
                    side * (gap / 2 + wall_r) + (arc_r - arc_r * np.cos(0)),
                ])
                static_obs.append(ObstacleStatic(pos=pos, radius=wall_r))

        # Curved segment (arc from angle 0 to angle 'span')
        for k in range(n_along + 1):
            phi = (k / n_along) * span
            # Inner wall (closer to arc center)
            inner_r = arc_r - gap / 2 - wall_r
            outer_r = arc_r + gap / 2 + wall_r
            for r_wall in (inner_r, outer_r):
                pos = np.array([
                    r_wall * np.sin(phi),
                    arc_r - r_wall * np.cos(phi),
                ])
                static_obs.append(ObstacleStatic(pos=pos, radius=wall_r))

        # Exit segment (straight after arc)
        exit_dir = np.array([np.cos(np.pi/2 - span), np.sin(np.pi/2 - span)])
        exit_normal = np.array([-exit_dir[1], exit_dir[0]])
        arc_end = np.array([arc_r * np.sin(span), arc_r - arc_r * np.cos(span)])
        post_length = 1.5
        n_post = int(post_length / spacing)
        for k in range(n_post):
            base = arc_end + (k + 0.5) * spacing * exit_dir
            for side in (+1, -1):
                pos = base + side * (gap / 2 + wall_r) * exit_normal
                static_obs.append(ObstacleStatic(pos=pos, radius=wall_r))

        # Start and goal
        start_center = np.array([-pre_length - 0.7, arc_r])
        goal = arc_end + (post_length + 0.7) * exit_dir

        # Face along corridor at start
        start_orientation = 0.0

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=start_orientation,
            static_obstacles=static_obs,
            dynamic_obstacles=[],
            metadata={
                "scenario": "curved_slot",
                "arc_radius": arc_r,
                "arc_span_deg": np.degrees(span),
                "gap_width": gap,
            },
        )
