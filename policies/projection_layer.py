"""
Feasibility-Preserving Projection Layer — Method A core contribution (C2).

Given the policy's raw affine intention z ∈ R^6 and the current sensed geometry,
project the decoded formation onto the feasible set so that every per-vehicle
reference is reachable/clear, while staying as close as possible to the intent.

We use the *minimal isotropic contraction*: find the largest gamma ∈ (0, 1] s.t.
the contracted offsets gamma·A(z)·d_i^0 keep the formation within the feasible
region, then apply it. This:
    - preserves the formation orientation / aspect ratio the policy chose,
    - has a closed form (differentiable), so gradients flow into the policy,
    - directly yields the "shrink to squeeze through the gap" behavior we want
      in the killer scenarios, and a feasibility guarantee before MPC.

Two mirrored implementations exist by design (same pattern as the affine decoder):
    - This torch layer: differentiable, width-based. Its feasibility spec is a
      free half-width w_allow perpendicular to the path (derivable from the
      LiDAR observation), so it can sit inside a differentiable policy / MPC
      pipeline and is used for the C2 ablation.
    - `policies.affine_decode_np.project_affine_offsets_np`: numpy, obstacle-
      exact. Runs inside FormationEnv during rollout (it has ground-truth
      obstacle geometry) and is what enforces feasibility during training.

The full linearized-QP variant (constraint per obstacle half-plane) is a strict
generalization; the isotropic contraction is its 1-D specialization along the
size axis and is what we validate in the pilot.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from policies.affine_policy import decode_affine_action


def perpendicular_extent(
    offsets: torch.Tensor,   # (B, N, 2) transformed offsets A(z)·d_i^0
    path_dir: torch.Tensor,  # (B, 2) unit path/goal direction
    eps: float = 1e-8,
) -> torch.Tensor:
    """Max |projection of any offset onto the cross-path axis| → (B,).

    This is the formation's half-extent perpendicular to the direction of
    travel — the quantity that must fit within a corridor's free half-width.
    """
    # Unit perpendicular n = R(90°) · path_dir = (-py, px)
    pd = path_dir / (path_dir.norm(dim=-1, keepdim=True) + eps)  # (B, 2)
    n = torch.stack([-pd[:, 1], pd[:, 0]], dim=-1)               # (B, 2)
    proj = torch.einsum("bnj,bj->bn", offsets, n)                # (B, N)
    return proj.abs().amax(dim=-1)                               # (B,)


class FeasibilityProjectionLayer(nn.Module):
    """Project a raw affine action onto the feasible set via isotropic shrink.

    Parameters
    ----------
    enable_projection : bool
        If False the layer is the identity (ablation: C2 off).
    theta_max, kappa_max : float
        Affine decode bounds (must match the policy / env).
    eps : float
        Numerical floor.
    """

    def __init__(
        self,
        enable_projection: bool = True,
        theta_max: float = 3.14159265 / 2,
        kappa_max: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.enable_projection = enable_projection
        self.theta_max = float(theta_max)
        self.kappa_max = float(kappa_max)
        self.eps = float(eps)

    def forward(
        self,
        z: torch.Tensor,                  # (B, 6) raw action in [-1, 1]
        formation_offsets: torch.Tensor,  # (N, 2) nominal d_i^0
        path_dir: torch.Tensor,           # (B, 2) unit path direction
        w_allow: torch.Tensor,            # (B,) or scalar — free half-width budget
        cfg,
    ):
        """Return (projected_offsets (B, N, 2), gamma (B,)).

        projected_offsets = gamma · A(z) · d_i^0, with
            gamma = clamp(w_allow / e_perp, max=1.0)
        so the contracted cross-path extent never exceeds w_allow.
        """
        offsets = decode_affine_action(
            z, formation_offsets, cfg,
            theta_max=self.theta_max, kappa_max=self.kappa_max,
        )  # (B, N, 2)

        B = offsets.shape[0]
        if not self.enable_projection:
            return offsets, torch.ones(B, device=offsets.device, dtype=offsets.dtype)

        e_perp = perpendicular_extent(offsets, path_dir, self.eps)  # (B,)
        if not torch.is_tensor(w_allow):
            w_allow = torch.as_tensor(
                float(w_allow), device=offsets.device, dtype=offsets.dtype
            ).expand(B)
        w_allow = w_allow.to(offsets.dtype).reshape(-1)

        gamma = torch.clamp(w_allow / (e_perp + self.eps), max=1.0)  # (B,)
        projected = gamma.view(B, 1, 1) * offsets
        return projected, gamma
