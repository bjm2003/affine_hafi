"""
Affine Formation Intention Policy — Method A core contribution (C1).

Parameterizes the team-level formation intention on the Aff(2) affine group:
    A(z) ∈ Aff(2) = ℝ² ⋊ GL(2)

Policy outputs 6-dim continuous action z ∈ [-1, 1]^6:
    z = (dx, dy, theta, s_x, s_y, kappa)

Decoding to affine transformation:
    - (dx, dy):        centroid subgoal direction (scale R0)
    - theta:           rotation angle, mapped to ±theta_max
    - (s_x, s_y):      anisotropic scale, mapped to [s_min, s_max] per axis
    - kappa:           shear coefficient, mapped to ±kappa_max

Formation offsets after transformation:
    D(z)_i = T(dx, dy) + R(theta) · S(s_x, s_y) · K(kappa) · d_i^0

where:
    R(theta) = [[cos θ, -sin θ],
                [sin θ,  cos θ]]
    S(s_x, s_y) = diag(s_x, s_y)
    K(kappa) = [[1, kappa],
                [0,     1]]

TODO:
    - Bounds: theta_max, s_min/s_max (from cfg.s_min, cfg.s_max), kappa_max
    - Consider parameterizing on Lie algebra 𝔞𝔣𝔣(2) and exp-mapping
    - Add curriculum wrapper: start with 3D (dx, dy, s_iso), gradually enable rotation, aniso, shear
"""

from __future__ import annotations
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class AffineActionHead(nn.Module):
    """6-head continuous action network for affine formation intention.

    latent_pi → [dx_head, dy_head, theta_head, sx_head, sy_head, kappa_head] → 6D
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        # 6 individual heads for interpretability
        self.dx_head = nn.Linear(hidden_dim, 1)
        self.dy_head = nn.Linear(hidden_dim, 1)
        self.theta_head = nn.Linear(hidden_dim, 1)
        self.sx_head = nn.Linear(hidden_dim, 1)
        self.sy_head = nn.Linear(hidden_dim, 1)
        self.kappa_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        h = self.trunk(latent_pi)
        z = torch.cat([
            self.dx_head(h),
            self.dy_head(h),
            self.theta_head(h),
            self.sx_head(h),
            self.sy_head(h),
            self.kappa_head(h),
        ], dim=-1)
        return z  # (batch, 6)


def decode_affine_action(
    z: torch.Tensor,        # (batch, 6) in [-1, 1]
    formation_offsets: torch.Tensor,  # (N, 2) nominal d_i^0
    cfg,
    theta_max: float = 3.14159 / 2,   # ±90° default (tune later)
    kappa_max: float = 0.5,
) -> torch.Tensor:
    """Decode z into per-vehicle formation offsets D(z)_i.

    Returns
    -------
    D_z : (batch, N, 2) transformed offsets
    """
    B = z.shape[0]
    N = formation_offsets.shape[0]
    device = z.device

    # 1. Unpack z
    theta = z[:, 2] * theta_max                     # (B,)
    s_x = cfg.s_min + (z[:, 3] + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)  # (B,)
    s_y = cfg.s_min + (z[:, 4] + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)  # (B,)
    kappa = z[:, 5] * kappa_max                     # (B,)

    # 2. Build 2x2 matrix per batch: A = R(theta) @ S(s_x, s_y) @ K(kappa)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1),
    ], dim=-2)  # (B, 2, 2)

    S = torch.zeros(B, 2, 2, device=device)
    S[:, 0, 0] = s_x
    S[:, 1, 1] = s_y

    K = torch.zeros(B, 2, 2, device=device)
    K[:, 0, 0] = 1.0
    K[:, 1, 1] = 1.0
    K[:, 0, 1] = kappa

    A = R @ S @ K  # (B, 2, 2)

    # 3. Apply to formation offsets
    #    D(z)_i = A @ d_i^0, then add translation later at env level (via subgoal)
    d0 = formation_offsets.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
    D_z = torch.einsum('bij,bnj->bni', A, d0)  # (B, N, 2)

    return D_z


class AffinePolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy with 6D affine action head.

    Inherits value/critic, action distribution, GAE, etc. from SB3.
    Only replaces action_net with AffineActionHead.

    TODO:
        - Override _build() to install AffineActionHead
        - Configure action distribution (DiagGaussian is fine for now)
        - Optionally add feasibility projection via projection_layer (see C2)
    """

    def __init__(self, *args, **kwargs):
        # Extract custom kwargs before super().__init__
        self.affine_theta_max = kwargs.pop("affine_theta_max", 3.14159 / 2)
        self.affine_kappa_max = kwargs.pop("affine_kappa_max", 0.5)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        latent_dim = self.mlp_extractor.latent_dim_pi
        self.action_net = AffineActionHead(latent_dim)
        # Rebuild optimizer to include new action_net params
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
