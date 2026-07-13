"""
Feasibility-Preserving Projection Layer — Method A core contribution (C2).

Given policy output z ∈ R^6 (raw affine intention), project onto the set of
feasible affine transformations for the current state s_t:

    z_hat = argmin_{z'} ||z' - z||_M^2
        s.t.  ∀i, D_i(z') ∈ F_i(s_t)   (each vehicle reference in its MPC feasible domain)

For pilot implementation, we approximate F_i(s_t) with a simple ball around the
current position:
    F_i(s_t) ≈ {p : ||p - p_i^{cur}|| ≤ H · v_max · dt_mpc}

where H is the MPC horizon. This ensures the reference is reachable within
one MPC horizon at max speed (necessary but not sufficient condition).

Later versions can use:
    - Velocity-space CBF (Ames et al.)
    - Free-space clustering from LiDAR
    - Learned feasibility indicator

Design principle: use differentiable QP (cvxpylayers) so gradient flows back
into the policy.

TODO:
    - Implement projection as differentiable QP
    - Prove Proposition 1 (feasibility guarantee under mild conditions)
    - Add ablation flag: enable_projection=False to compare against no-projection baseline
"""

from __future__ import annotations
import torch
import torch.nn as nn


class FeasibilityProjectionLayer(nn.Module):
    """Project raw affine action z ∈ R^6 onto feasibility set.

    Simplified pilot version: box constraints on each dim + soft penalty for
    out-of-reachable-set references. Full version uses cvxpylayers.

    Parameters
    ----------
    enable_projection : bool
        If False, layer is identity (for ablation).
    reachable_radius : float
        H · v_max · dt_mpc, used as feasibility ball radius.
    """

    def __init__(
        self,
        enable_projection: bool = True,
        reachable_radius: float = 0.1 * 10 * 0.1,  # v_max=0.1, H=10, dt=0.1 → 0.1m
    ):
        super().__init__()
        self.enable_projection = enable_projection
        self.reachable_radius = reachable_radius

    def forward(
        self,
        z: torch.Tensor,                     # (B, 6) raw action
        current_positions: torch.Tensor,     # (B, N, 2)
        subgoal_center: torch.Tensor,        # (B, 2)
        formation_offsets: torch.Tensor,     # (N, 2)
        cfg,
    ) -> torch.Tensor:
        """Return projected z_hat ∈ R^6.

        Pilot version: identity when enable_projection=False; light box
        clipping otherwise. Full QP version is a TODO.
        """
        if not self.enable_projection:
            return z

        # TODO: implement differentiable QP
        # For pilot, just clip to [-1, 1] (already enforced by action space, so no-op)
        z_hat = torch.clamp(z, -1.0, 1.0)

        # TODO: add distance-based soft projection:
        #   for each vehicle i:
        #     p_i^ref = subgoal_center + D_i(z_hat)
        #     if ||p_i^ref - current_positions[i]|| > reachable_radius:
        #       scale down (s_x, s_y, kappa) toward identity
        return z_hat


def build_qp_projection_matrix(
    formation_offsets: torch.Tensor,   # (N, 2)
    reachable_radius: float,
):
    """Placeholder for constructing the QP constraint matrices.

    The QP is:
        min ||z' - z||^2
        s.t.  ||A(z') · d_i^0 + subgoal - p_i^cur|| ≤ reachable_radius,  for all i

    This is non-convex in z' due to the trigonometric terms in R(theta).
    Options:
        (a) Linearize around z (SQP-style, differentiable)
        (b) Use SDP relaxation
        (c) Use a small neural net to approximate the projection

    Recommend (a) for pilot — one SQP iteration is a linear QP that
    cvxpylayers can handle.

    TODO: implement.
    """
    raise NotImplementedError("QP projection to be implemented in M2")
