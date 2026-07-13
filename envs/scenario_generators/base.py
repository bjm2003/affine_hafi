"""
Scenario generator base class and data structures.

Every scenario returns a `ScenarioInstance` describing:
    - start_center: (2,)   team centroid at t=0 (goal will be offset from here)
    - goal:         (2,)   target position
    - static_obstacles:  list of (pos, radius)
    - dynamic_obstacles: list of (pos, radius, velocity)
    - arena_bounds:      optional (xmin, xmax, ymin, ymax) for MPC arena constraint

Each generator is a callable class implementing `sample(rng, cfg) -> ScenarioInstance`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ObstacleStatic:
    pos: np.ndarray            # (2,)
    radius: float


@dataclass
class ObstacleDynamic:
    pos: np.ndarray            # (2,)
    radius: float
    velocity: np.ndarray       # (2,)


@dataclass
class ScenarioInstance:
    """A concrete sampled scenario. Everything the Env needs to reset."""
    start_center: np.ndarray                                # (2,)
    goal: np.ndarray                                         # (2,)
    start_orientation: float = 0.0                           # initial team heading (rad)
    static_obstacles: List[ObstacleStatic] = field(default_factory=list)
    dynamic_obstacles: List[ObstacleDynamic] = field(default_factory=list)
    arena_bounds: Optional[Tuple[float, float, float, float]] = None  # (xmin,xmax,ymin,ymax)
    inject_schedule: List[Tuple[int, ObstacleDynamic]] = field(default_factory=list)  # (step, obs) to inject mid-episode
    metadata: dict = field(default_factory=dict)             # e.g. {"scenario": "curved_slot", "gap_width": 0.7}


class BaseScenario:
    """Base class for scenario generators."""

    name: str = "base"

    def sample(self, rng: np.random.Generator, cfg) -> ScenarioInstance:
        raise NotImplementedError

    # ------------------------------------------------------------------
    #  Common utilities
    # ------------------------------------------------------------------
    @staticmethod
    def sample_start_goal_pair(
        rng: np.random.Generator,
        cfg,
        min_dist: float = None,
        max_dist: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample start_center and goal with distance in cfg.start_distance_range."""
        lo, hi = cfg.start_distance_range if hasattr(cfg, "start_distance_range") else (6.0, 10.0)
        if min_dist is not None:
            lo = min_dist
        if max_dist is not None:
            hi = max_dist
        dist = rng.uniform(lo, hi)
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        # Center start at origin, goal in random direction
        # (Later scenarios override to place in structured world)
        start_center = np.zeros(2)
        goal = start_center + dist * direction
        return start_center, goal

    @staticmethod
    def sample_static_obstacles_uniform(
        rng: np.random.Generator,
        cfg,
        n: int,
        world_half: float,
        radius_range: Tuple[float, float] = None,
        exclude_zones: List[Tuple[np.ndarray, float]] = None,
    ) -> List[ObstacleStatic]:
        """Sample n static obstacles uniformly in [-world_half, world_half]^2,
        avoiding exclude_zones = [(center, radius), ...]."""
        if radius_range is None:
            radius_range = getattr(cfg, "static_obs_radius_range", (0.10, 0.30))
        exclude_zones = exclude_zones or []

        obs = []
        attempts = 0
        max_attempts = n * 50
        while len(obs) < n and attempts < max_attempts:
            attempts += 1
            pos = rng.uniform(-world_half, world_half, size=2)
            radius = float(rng.uniform(*radius_range))
            # Reject if in excluded zone
            ok = True
            for ex_pos, ex_r in exclude_zones:
                if np.linalg.norm(pos - ex_pos) < ex_r + radius:
                    ok = False
                    break
            # Reject if overlaps existing
            for o in obs:
                if np.linalg.norm(pos - o.pos) < o.radius + radius + 0.15:
                    ok = False
                    break
            if ok:
                obs.append(ObstacleStatic(pos=pos, radius=radius))
        return obs

    @staticmethod
    def sample_dynamic_obstacles(
        rng: np.random.Generator,
        cfg,
        n: int,
        world_half: float,
        radius_range: Tuple[float, float] = None,
        speed_range: Tuple[float, float] = None,
        exclude_zones: List[Tuple[np.ndarray, float]] = None,
    ) -> List[ObstacleDynamic]:
        if radius_range is None:
            radius_range = getattr(cfg, "obs_radius_range", (0.10, 0.30))
        if speed_range is None:
            spd_max = getattr(cfg, "dynamic_obs_speed", 0.20)
            speed_range = (0.10, spd_max)
        exclude_zones = exclude_zones or []

        obs = []
        attempts = 0
        max_attempts = n * 50
        while len(obs) < n and attempts < max_attempts:
            attempts += 1
            pos = rng.uniform(-world_half, world_half, size=2)
            radius = float(rng.uniform(*radius_range))
            speed = float(rng.uniform(*speed_range))
            angle = rng.uniform(-np.pi, np.pi)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            ok = True
            for ex_pos, ex_r in exclude_zones:
                if np.linalg.norm(pos - ex_pos) < ex_r + radius:
                    ok = False
                    break
            if ok:
                obs.append(ObstacleDynamic(pos=pos, radius=radius, velocity=vel))
        return obs
