"""
Numpy port of policies/affine_policy.py::decode_affine_action.

Called from envs/formation_env.py (numpy vectorized environment)
to decode a 6D affine action into per-vehicle formation offsets, without
introducing a torch dependency at the Env boundary (helps SubprocVecEnv
pickling and vec-env throughput).

Kept structurally identical to the torch version so unit tests can compare
elementwise:
    A(z) = R(theta) · S(s_x, s_y) · K(kappa)
    D(z)_i = A(z) · d_i^0

The env is then responsible for adding the subgoal translation term:
    p_i^ref = p_c^ref + D(z)_i
"""

from __future__ import annotations
import numpy as np


def decode_affine_action_np(
    z: np.ndarray,                    # (6,) in [-1, 1]
    formation_offsets: np.ndarray,    # (N, 2) nominal d_i^0
    cfg,
    theta_max: float = np.pi / 2,     # ±90° default
    kappa_max: float = 0.5,
) -> np.ndarray:
    """Decode z into per-vehicle formation offsets D(z)_i.

    Parameters
    ----------
    z : (6,) array
        (dx, dy, theta, s_x, s_y, kappa), each in [-1, 1].
        Only components 2..5 (theta, s_x, s_y, kappa) affect this function;
        (dx, dy) are handled by the caller as a translation.
    formation_offsets : (N, 2) array
        Nominal offsets d_i^0 from team centroid (unscaled formation template).
    cfg : Config
        Uses cfg.s_min, cfg.s_max for scale bounds.
    theta_max : float
        Rotation bound (±rad).
    kappa_max : float
        Shear coefficient bound (±dimensionless).

    Returns
    -------
    D_z : (N, 2) array
        Transformed offsets (no translation added).
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    assert z.shape == (6,), f"Expected 6D action, got shape {z.shape}"

    # Unpack z (clip to enforce action space bounds even if policy escapes)
    theta = float(np.clip(z[2], -1.0, 1.0)) * theta_max
    s_x = cfg.s_min + (float(np.clip(z[3], -1.0, 1.0)) + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)
    s_y = cfg.s_min + (float(np.clip(z[4], -1.0, 1.0)) + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)
    kappa = float(np.clip(z[5], -1.0, 1.0)) * kappa_max

    # Build 2x2 affine matrix A = R @ S @ K
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    S = np.array([[s_x, 0.0], [0.0, s_y]], dtype=np.float64)
    K = np.array([[1.0, kappa], [0.0, 1.0]], dtype=np.float64)
    A = R @ S @ K  # (2, 2)

    # Apply to formation offsets: D(z)_i = A @ d_i^0
    D_z = formation_offsets @ A.T  # (N, 2)
    return D_z


def effective_isotropic_scale(z: np.ndarray, cfg) -> float:
    """Convenience: return an "effective scale" scalar for reward + logging.

    We use the geometric mean of s_x and s_y so ablations (w/o anisotropic)
    reduce cleanly to the HAFI 1D scale semantics.
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    s_x = cfg.s_min + (float(np.clip(z[3], -1.0, 1.0)) + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)
    s_y = cfg.s_min + (float(np.clip(z[4], -1.0, 1.0)) + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)
    return float(np.sqrt(s_x * s_y))
