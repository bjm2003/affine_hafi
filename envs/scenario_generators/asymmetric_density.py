"""KILLER SCENARIO 3 — Asymmetric Density Field

一侧障碍密集 (静态 + 动态), 另一侧空旷。

预测: 均匀 scale 无法利用一侧空旷;
      我们 aniso-scale (s_x, s_y 不等) + delta_y 偏移 靠向空旷侧。

关键差异化: 这是 He & Jing 2025 (arXiv 2508.02289) 的经典 motivation 场景;
           但他们是 classical control, 无人做 learning 版本。
"""

from __future__ import annotations
import numpy as np
from .base import BaseScenario, ScenarioInstance


class AsymmetricDensityScenario(BaseScenario):
    name = "asymmetric_density"

    def __init__(
        self,
        dense_side_n_static: int = 10,
        sparse_side_n_static: int = 2,
        density_asymmetry_offset: float = 1.5,   # perpendicular offset between dense and sparse
    ):
        self.dense_n = dense_side_n_static
        self.sparse_n = sparse_side_n_static
        self.asym = density_asymmetry_offset

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        world_half = cfg.world_size / 2.0

        # Fixed axis (along +x for reproducibility) — training envs randomize orientation
        theta = float(rng.uniform(-np.pi, np.pi))
        axis = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])

        dist_along = float(rng.uniform(6.0, 8.0))
        start_center = -0.5 * dist_along * axis
        goal = 0.5 * dist_along * axis

        # Dense side: obstacles clustered on one side of the corridor
        dense_center = 0.0 * axis + self.asym * normal      # +normal side
        sparse_center = 0.0 * axis - self.asym * normal     # -normal side

        exclude_start = [(start_center, 1.5), (goal, 1.0)]

        # Custom sampling: constrain to a rectangle near path
        static_obs = []
        cluster_half = 2.5   # rectangle 5m along axis
        cluster_width = 1.0  # rectangle 2m perpendicular

        def sample_in_rect(center, n, radius_range):
            out = []
            attempts = 0
            while len(out) < n and attempts < 100 * n:
                attempts += 1
                a = rng.uniform(-cluster_half, cluster_half)
                b = rng.uniform(-cluster_width / 2, cluster_width / 2)
                pos = center + a * axis + b * normal
                r = float(rng.uniform(*radius_range))
                ok = True
                for e_pos, e_r in exclude_start:
                    if np.linalg.norm(pos - e_pos) < e_r + r:
                        ok = False; break
                for o in out:
                    if np.linalg.norm(pos - o.pos) < o.radius + r + 0.1:
                        ok = False; break
                if ok:
                    from .base import ObstacleStatic
                    out.append(ObstacleStatic(pos=pos, radius=r))
            return out

        radius_range = getattr(cfg, "static_obs_radius_range", (0.10, 0.30))
        static_obs.extend(sample_in_rect(dense_center, self.dense_n, radius_range))
        static_obs.extend(sample_in_rect(sparse_center, self.sparse_n, radius_range))

        # No dynamic in this scenario (isolate asymmetric geometry effect)
        dynamic_obs = []

        return ScenarioInstance(
            start_center=start_center,
            goal=goal,
            start_orientation=theta,
            static_obstacles=static_obs,
            dynamic_obstacles=dynamic_obs,
            metadata={
                "scenario": "asymmetric_density",
                "dense_n": self.dense_n,
                "sparse_n": self.sparse_n,
                "asymmetry_offset": self.asym,
                "dense_side_normal": normal.tolist(),
            },
        )
