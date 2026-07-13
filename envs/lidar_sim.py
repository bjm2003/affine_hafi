"""
Simulated LiDAR for FormationEnv.

Given leader position, orientation, and world obstacles (static + dynamic),
produce a (n_lidar_rays,) range array in [0, lidar_max_dist].

Rays are indexed CCW starting from world +x (matching deploy package
commander_node.py:_lidar_angles). LiDAR is body-frame, so caller must derotate
back to world frame if downstream needs world angles.

This is a pure numpy simulator (no ROS, no Gazebo). Fast enough for training
with 12 parallel envs.
"""

from __future__ import annotations
import numpy as np
from typing import List

# Type imports (used only for annotations to avoid circular)
from .scenario_generators.base import ObstacleStatic, ObstacleDynamic


def simulate_lidar(
    origin: np.ndarray,          # (2,) LiDAR sensor position (world frame)
    yaw: float,                  # LiDAR body orientation (rad, world frame +x = 0)
    static_obs: List[ObstacleStatic],
    dynamic_obs: List[ObstacleDynamic],
    n_rays: int,
    max_dist: float,
    teammate_positions: np.ndarray = None,  # (N-1, 2) teammate positions to filter out
    teammate_radius: float = 0.20,
) -> np.ndarray:
    """Return (n_rays,) LiDAR range array.

    Ray k (k=0..n_rays-1) points in body-frame angle:
        theta_body_k = 2*pi * k / n_rays
    In world frame:
        theta_world_k = yaw + theta_body_k

    Returned distances are along the ray in body frame (same as world frame magnitude).
    """
    ranges = np.full(n_rays, max_dist, dtype=np.float64)

    body_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    world_angles = (body_angles + yaw) % (2 * np.pi)
    cos_w = np.cos(world_angles)
    sin_w = np.sin(world_angles)

    all_obs = []
    for o in static_obs:
        all_obs.append((o.pos, o.radius))
    for o in dynamic_obs:
        all_obs.append((o.pos, o.radius))

    # Teammates as filterable obstacles (marked separately)
    teammate_list = []
    if teammate_positions is not None:
        for tp in teammate_positions:
            teammate_list.append((np.asarray(tp), teammate_radius))

    for k in range(n_rays):
        d_ray = cos_w[k]
        # Cast against each obstacle (ray-circle intersection)
        min_hit = max_dist
        for obs_pos, obs_r in all_obs + teammate_list:
            hit = _ray_circle_hit(
                origin, np.array([cos_w[k], sin_w[k]]),
                obs_pos, obs_r,
            )
            if hit is not None and hit < min_hit:
                min_hit = hit
        ranges[k] = min_hit

    return ranges


def _ray_circle_hit(
    origin: np.ndarray, direction: np.ndarray,
    center: np.ndarray, radius: float,
) -> float | None:
    """Return hit distance ≥0 along ray, or None if miss.

    Solves ||origin + t * direction - center||^2 = radius^2 for t ≥ 0.
    """
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)
    if c <= 0:
        # origin inside circle → hit at t=0 (return 0.0)
        return 0.0
    disc = b * b - c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc
    # Nearest positive
    if t1 > 0:
        return t1
    if t2 > 0:
        return t2
    return None


def lidar_to_spatial_dirs(
    lidar_ranges: np.ndarray,     # (n_rays,) raw (in body frame after derotate)
    n_dirs: int,
    max_dist: float,
) -> np.ndarray:
    """Min-pool n_rays into n_dirs sectors (world frame, matching HAFI obs).

    Returns (n_dirs,) normalized distances in [0, 1].
    Caller is responsible for derotating to world frame BEFORE calling this.
    """
    n_rays = lidar_ranges.shape[0]
    # Sector d covers world angles [d * 2pi/n_dirs - pi/n_dirs, d * 2pi/n_dirs + pi/n_dirs]
    ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    spatial = np.ones(n_dirs, dtype=np.float32)
    half_width = np.pi / n_dirs
    for d in range(n_dirs):
        center = d * (2.0 * np.pi / n_dirs)
        rel = ray_angles - center
        rel = (rel + np.pi) % (2.0 * np.pi) - np.pi
        mask = np.abs(rel) < half_width
        if mask.any():
            spatial[d] = float(np.clip(lidar_ranges[mask].min() / max_dist, 0.0, 1.0))
    return spatial


def derotate_lidar(sensor_ranges: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate body-frame lidar ranges into world-frame indexed ranges.

    Matches deploy commander_node.py:_derotate_lidar.
    """
    n = sensor_ranges.shape[0]
    world_angles = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
    sensor_angles_needed = (world_angles - yaw) % (2.0 * np.pi)
    src = sensor_angles_needed * n / (2.0 * np.pi)
    i0 = np.floor(src).astype(int) % n
    i1 = (i0 + 1) % n
    w1 = src - np.floor(src)
    w0 = 1.0 - w1
    return (w0 * sensor_ranges[i0] + w1 * sensor_ranges[i1]).astype(np.float64)
