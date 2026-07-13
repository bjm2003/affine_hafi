"""KILLER SCENARIO 4 — Sudden Interior Injection

在编队通过路径中期, 障碍**注入在编队几何中心位置**。

预测: HAFI 只能 σ 缩最小仍撞;
      STAF split 触发时机来不及;
      Ours: shear + rotation 瞬间"扭"避开中心。

关键差异化: 现有方法应对突发注入都需要预判或快速切换,
           我们的 continuous manifold 具备毫秒级 policy 响应。
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance, ObstacleDynamic


class InteriorInjectionScenario(BaseScenario):
    name = "interior_injection"

    def __init__(
        self,
        inject_at_progress: float = 0.5,     # inject at 50% of path
        inject_offset_from_center: float = 0.0,   # 0 = right at team centroid
        inject_radius: float = 0.20,
        inject_speed: float = 0.15,           # slow-drift injection
    ):
        self.inject_at = inject_at_progress
        self.inject_offset = inject_offset_from_center
        self.inject_r = inject_radius
        self.inject_speed = inject_speed

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        world_half = cfg.world_size / 2.0

        theta = float(rng.uniform(-np.pi, np.pi))
        axis = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])

        dist_along = float(rng.uniform(6.0, 8.0))
        start_center = -0.5 * dist_along * axis
        goal = 0.5 * dist_along * axis

        # Baseline scene: a few static obs like Level 1
        exclude = [(start_center, 1.5), (goal, 1.0)]
        static_obs = self.sample_static_obstacles_uniform(
            rng, cfg, n=int(rng.integers(2, 4)),
            world_half=world_half, exclude_zones=exclude,
        )

        # Schedule injection at inject_at_progress step
        # Assuming episode takes ~ max_episode_steps to complete
        max_steps = getattr(cfg, "max_episode_steps", 150)
        inject_step = int(self.inject_at * max_steps)

        # Injection position: expected team centroid at inject_step
        expected_center = start_center + self.inject_at * (goal - start_center) \
                        + self.inject_offset * normal
        # Injection velocity: drift toward team's motion axis (crossing path)
        drift_angle = rng.uniform(-np.pi, np.pi)
        inject_vel = self.inject_speed * np.array([
            np.cos(drift_angle), np.sin(drift_angle),
        ])

        inject_obs = ObstacleDynamic(
            pos=expected_center,
            radius=self.inject_r,
            velocity=inject_vel,
        )
        inject_schedule = [(inject_step, inject_obs)]

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=theta,
            static_obstacles=static_obs,
            dynamic_obstacles=[],
            inject_schedule=inject_schedule,
            metadata={
                "scenario": "interior_injection",
                "inject_step": inject_step,
                "inject_progress": self.inject_at,
                "inject_pos": expected_center.tolist(),
            },
        )
