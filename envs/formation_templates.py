"""
Formation offset templates (matches deploy package commander_node.py:157-177).

Given N vehicles and edge length d_form (nominal pairwise distance in the
unscaled formation), returns (N, 2) array of offsets from team centroid.

Shapes:
    N=2 : line perpendicular to +x
    N=3 : equilateral triangle, leader (idx 0) at +x
    N=4 : square
    N=5+: regular N-gon (centered at origin)
"""

from __future__ import annotations
import numpy as np


def build_formation_offsets(n_vehicles: int, d_form: float) -> np.ndarray:
    d = float(d_form)
    n = int(n_vehicles)

    if n == 2:
        return np.array([
            [0.0, d / 2],
            [0.0, -d / 2],
        ], dtype=np.float64)

    if n == 3:
        # Equilateral triangle, leader (idx 0) on +x axis
        # side length = d, centroid at origin, apex distance = d / sqrt(3)
        return np.array([
            [d * np.sqrt(3) / 3, 0.0],
            [-d * np.sqrt(3) / 6, d / 2],
            [-d * np.sqrt(3) / 6, -d / 2],
        ], dtype=np.float64)

    # Regular N-gon (N ≥ 4), circumradius = d / (2 sin(pi/N))
    R = d / (2.0 * np.sin(np.pi / n))
    offsets = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        offsets[i] = [R * np.cos(angle), R * np.sin(angle)]
    # Center at origin (guaranteed by symmetry, but ensure)
    offsets -= offsets.mean(axis=0)
    return offsets


def nominal_pairwise_distances(offsets: np.ndarray) -> np.ndarray:
    """(N, N) matrix of unscaled pairwise distances between formation slots."""
    n = offsets.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = float(np.linalg.norm(offsets[i] - offsets[j]))
    return d
