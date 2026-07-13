"""
Reward function for FormationEnv, aligned with HAFI paper:

    r_t = w_progress · r_progress
        + w_goal · r_goal
        + w_scale · r_scale
        + w_delta_s · r_delta_s
        + w_collision · r_collision
        + w_formation · r_formation
        + w_time · r_time
        + [optional] w_ent · r_formation_entropy  ← C3 fallback

Component definitions (from paper Eq. 1):
    r_progress = max(d_{t-1} - d_t, -delta_clip)
    r_goal     = 1 if d_t < eps_g else 0
    r_scale    = 0 if scale <= 1 else -(scale - 1)^2         (penalize expansion)
    r_delta_s  = -|scale_t - scale_{t-1}|                    (smooth scale changes)
    r_collision= -1 if collision else 0
    r_formation= -deviation of pairwise distances from scale-adjusted nominal
    r_time     = -1                                          (small time penalty)

For affine action (Ours), formation entropy regularizer (optional fallback):
    r_formation_entropy = -log det(D(z)^T D(z) + eps · I)    (prevent topology collapse)

TODO:
    - Implement each component with vectorized numpy
    - Add w_scale_corridor variant for corridor scenarios
    - Add proximity penalty option (w_proximity)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


def compute_reward(
    positions: np.ndarray,     # (N, 2) current vehicle positions
    prev_center: np.ndarray,   # (2,) previous team centroid
    goal: np.ndarray,          # (2,)
    current_scale: float,
    prev_scale: float,
    formation_offsets: np.ndarray,  # (N, 2) nominal offsets
    collision: bool,
    cfg,                       # Config
    formation_manifold_matrix: np.ndarray = None,  # (N, 2) for entropy reg (optional)
    weights: Dict[str, float] = None,
) -> Tuple[float, Dict[str, float]]:
    """Return (total_reward, component_dict)."""
    # TODO: implement component-by-component
    # For now, return zero to allow env skeleton to run
    components = {
        "progress": 0.0,
        "goal": 0.0,
        "scale": 0.0,
        "delta_s": 0.0,
        "collision": 0.0,
        "formation": 0.0,
        "time": 0.0,
    }
    total = sum(components.values())
    return total, components


def formation_entropy_reg(
    affine_offsets: np.ndarray,  # (N, 2) offsets after affine transform
    eps: float = 1e-4,
) -> float:
    """Compute -log det(D^T D + eps I) to prevent topology collapse.

    D = affine_offsets ∈ R^{N x 2}
    Returns a positive value; larger means more collapsed (bad).
    Use as a reward penalty (subtract from reward).
    """
    D = affine_offsets
    gram = D.T @ D + eps * np.eye(D.shape[1])
    sign, logdet = np.linalg.slogdet(gram)
    return -float(logdet) if sign > 0 else float("inf")
