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
    affine_action: Optional[np.ndarray] = None,   # (6,) raw affine z (for deform reg)
    clearance: Optional[float] = None,            # free space around formation (m)
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

    # ============ Optional: clearance-modulated deformation regularizer ============
    # Penalize UNNEEDED deformation: anisotropy/shear/rotation are cheap in tight
    # spots (clearance small) but should collapse to isotropy in open space
    # (clearance large). Directly attacks the observed degenerate saturation where
    # the policy pins anisotropy/rotation at the action-space corner everywhere.
    w_deform = float(getattr(cfg, "w_deform_reg", 0.0))
    if w_deform > 0.0 and affine_action is not None and clearance is not None:
        components["deform_reg"] = -w_deform * deformation_penalty(
            affine_action, float(clearance), cfg,
        )

    total = float(sum(components.values()))
    return total, components


def deformation_penalty(z: np.ndarray, clearance: float, cfg) -> float:
    """Non-negative penalty for deforming away from isotropy, gated by clearance.

    Decodes the affine action to (theta, s_x, s_y, kappa) using the same formula
    as policies/affine_decode_np.decode_affine_action_np, normalizes each free
    DOF to [0, 1], and scales the squared magnitude by a clearance gate:

        gate = clip((clearance - tight) / (open - tight), 0, 1)
             = 0 when clearance <= tight  (tight passage → deformation allowed)
             = 1 when clearance >= open   (open space   → full penalty)

    Returns 0 when the policy stays isotropic (s_x==s_y, kappa==0, theta==0),
    regardless of clearance.
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    theta = float(np.clip(z[2], -1.0, 1.0)) * float(cfg.affine_theta_max)
    s_x = cfg.s_min + (float(np.clip(z[3], -1.0, 1.0)) + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)
    s_y = cfg.s_min + (float(np.clip(z[4], -1.0, 1.0)) + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)
    kappa = float(np.clip(z[5], -1.0, 1.0)) * float(cfg.affine_kappa_max)

    aniso = abs(s_x - s_y) / max(cfg.s_max - cfg.s_min, 1e-6)
    shear = abs(kappa) / max(float(cfg.affine_kappa_max), 1e-6)
    rot = abs(theta) / max(float(cfg.affine_theta_max), 1e-6)
    deform = aniso ** 2 + shear ** 2 + rot ** 2

    tight = float(cfg.deform_clearance_tight)
    open_c = float(cfg.deform_clearance_open)
    gate = (clearance - tight) / max(open_c - tight, 1e-6)
    gate = float(np.clip(gate, 0.0, 1.0))
    return gate * deform


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
