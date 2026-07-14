"""
Reward function for FormationEnv, aligned with HAFI paper Eq. 1.

    r_t = w_prog · r_prog + w_goal · r_goal + w_scale · r_scale + w_dS · r_dS
        + w_coll · r_coll + w_form · r_form + w_time · r_time
        [+ w_ent · r_entropy]   ← optional (Ours affine, Component 3 fallback)

Component definitions:
    r_prog  = max(d_{t-1} - d_t, -delta_clip)                (progress toward goal)
    r_goal  = 1 if d_t < eps_g else 0                        (indicator on success)
    r_scale = -(scale - 1)^2  if scale > 1 else 0            (asymmetric: only over-expansion)
    r_dS    = -|scale_t - scale_{t-1}|                       (smooth scale changes)
    r_coll  = -1 if collision else 0                         (hard indicator)
    r_form  = -mean_{i<j} (||p_i - p_j|| - scale · d_{ij}^0)^2
                                                             (pairwise formation error)
    r_time  = -1                                             (constant time penalty)

Corridor override:
    In corridor family scenarios (corridor, s_corridor, z_corridor, curved_slot),
    w_scale is replaced with the stronger w_scale_corridor (config.py: 1.5 → 8.0).

Optional formation entropy regularization (for Method A affine ablation):
    r_entropy = -log det(D(z)^T D(z) + eps · I)
    (penalize configurations where formation collapses onto a lower-dim subspace)

Return:
    (total_reward: float, components: dict[str, float])
    components dict contains the WEIGHTED value of each term (post-multiplication)
    so wandb can log the contribution of each directly.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np


CORRIDOR_FAMILY = {"corridor", "s_corridor", "z_corridor", "curved_slot"}


def compute_reward(
    positions: np.ndarray,       # (N, 2) current vehicle positions AFTER step
    prev_center: np.ndarray,     # (2,)   previous team centroid (BEFORE step)
    goal: np.ndarray,            # (2,)
    current_scale: float,
    prev_scale: float,
    nominal_dists: np.ndarray,   # (N, N) unscaled pairwise distances
    collision: bool,
    cfg,                         # Config
    scenario_name: str = "open",
    affine_offsets: Optional[np.ndarray] = None,  # (N, 2) transformed offsets (for entropy)
    entropy_weight: float = 0.0,
    entropy_eps: float = 1e-4,
) -> Tuple[float, Dict[str, float]]:
    """Return (total_reward, weighted_component_dict)."""

    # ============ 1. Distance to goal (progress) ============
    center = positions.mean(axis=0)
    prev_dist = float(np.linalg.norm(prev_center - goal))
    curr_dist = float(np.linalg.norm(center - goal))
    progress_raw = max(prev_dist - curr_dist, -float(cfg.delta_clip))

    # ============ 2. Goal indicator ============
    at_goal = curr_dist < cfg.goal_tolerance
    goal_raw = 1.0 if at_goal else 0.0

    # ============ 3. Scale penalty (asymmetric) ============
    if current_scale > 1.0:
        scale_raw = -(current_scale - 1.0) ** 2
    else:
        scale_raw = 0.0

    # Corridor scenarios use stronger scale weight
    is_corridor = scenario_name in CORRIDOR_FAMILY
    w_scale_active = float(cfg.w_scale_corridor) if is_corridor else float(cfg.w_scale)

    # ============ 4. Smooth scale change ============
    delta_s_raw = -abs(current_scale - prev_scale)

    # ============ 5. Collision hard penalty ============
    collision_raw = -1.0 if collision else 0.0

    # ============ 6. Formation error (scale-adjusted) ============
    N = positions.shape[0]
    form_err_sq = 0.0
    n_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            actual = float(np.linalg.norm(positions[i] - positions[j]))
            target = float(current_scale) * float(nominal_dists[i, j])
            form_err_sq += (actual - target) ** 2
            n_pairs += 1
    formation_raw = -(form_err_sq / max(n_pairs, 1))

    # ============ 7. Time penalty ============
    time_raw = -1.0

    # ============ Weighted components ============
    components = {
        "progress": float(cfg.w_progress) * progress_raw,
        "goal": float(cfg.w_goal) * goal_raw,
        "scale": w_scale_active * scale_raw,
        "delta_s": float(cfg.w_delta_s) * delta_s_raw,
        "collision": float(cfg.w_collision) * collision_raw,
        "formation": float(cfg.w_formation) * formation_raw,
        "time": float(cfg.w_time) * time_raw,
    }

    # ============ Optional: formation entropy regularization ============
    if entropy_weight > 0.0 and affine_offsets is not None:
        ent = formation_entropy_reg(affine_offsets, eps=entropy_eps)
        # Note: formation_entropy_reg returns POSITIVE for collapsed configs;
        # penalize by subtracting from reward
        components["entropy"] = -float(entropy_weight) * ent

    total = float(sum(components.values()))
    return total, components


def formation_entropy_reg(
    affine_offsets: np.ndarray,  # (N, 2)
    eps: float = 1e-4,
) -> float:
    """Compute -log det(D^T D + eps I) as a positive scalar.

    Returns larger value for MORE collapsed configurations (e.g. all points
    on a line → det → 0 → -log(...) → +inf).

    Use as a REWARD PENALTY: subtract entropy_weight * this from reward,
    so policy is discouraged from collapsing formation into a lower-dim
    subspace.
    """
    D = np.asarray(affine_offsets, dtype=np.float64)
    gram = D.T @ D + eps * np.eye(D.shape[1])
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        return float("inf")
    return -float(logdet)
