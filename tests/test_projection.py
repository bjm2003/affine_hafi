"""
Unit tests for the feasibility-preserving projection layer (Method A, C2).

Covers:
    - numpy env-side projection (project_affine_offsets_np): no-op when feasible,
      shrinks to clear obstacles, respects arena bounds, monotone scaling.
    - torch differentiable layer (FeasibilityProjectionLayer): identity when
      disabled, cross-path width limiting, gradient flow.
"""

from __future__ import annotations
import numpy as np
import pytest
import torch

from config import Config
from policies.affine_decode_np import (
    decode_affine_action_np, project_affine_offsets_np,
)
from policies.projection_layer import (
    FeasibilityProjectionLayer, perpendicular_extent,
)


@pytest.fixture
def cfg():
    return Config()


# ---------------------------------------------------------------------------
#  numpy env-side projection
# ---------------------------------------------------------------------------
def test_np_projection_noop_when_no_obstacles():
    offsets = np.array([[0.5, 0.3], [-0.4, 0.2], [0.1, -0.5]])
    subgoal = np.array([2.0, 0.0])
    out, gamma, active = project_affine_offsets_np(
        offsets, subgoal, obstacles=[], vehicle_radius=0.18, d_safe=0.15,
    )
    assert gamma == 1.0
    assert active is False
    assert np.allclose(out, offsets)


def test_np_projection_shrinks_to_clear_obstacle():
    # One vehicle reference lands right on an obstacle → must shrink.
    offsets = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    subgoal = np.array([0.0, 0.0])
    # Obstacle exactly where vehicle 0's reference would be at gamma=1.
    obstacles = [(np.array([1.0, 0.0]), 0.3)]
    out, gamma, active = project_affine_offsets_np(
        offsets, subgoal, obstacles,
        vehicle_radius=0.18, d_safe=0.15, clearance=0.05,
        gamma_min=0.1, gamma_steps=8,
    )
    assert active is True
    assert gamma < 1.0
    # Projected offsets are an isotropic scaling of the originals.
    assert np.allclose(out, gamma * offsets)
    # Every projected reference now clears the obstacle by the required margin.
    min_clear = 0.18 + 0.15 + 0.05
    refs = subgoal[None, :] + out
    for ref in refs:
        assert np.linalg.norm(ref - obstacles[0][0]) >= obstacles[0][1] + min_clear - 1e-9


def test_np_projection_respects_arena_bounds():
    # Reference pushed outside a tight arena → shrink until inside.
    offsets = np.array([[3.0, 0.0], [-0.3, 0.0], [0.0, 0.3]])
    subgoal = np.array([0.0, 0.0])
    arena = (-1.0, 1.0, -1.0, 1.0)  # xmax=1.0, vehicle_radius inflation
    out, gamma, active = project_affine_offsets_np(
        offsets, subgoal, obstacles=[], vehicle_radius=0.18, d_safe=0.15,
        arena_bounds=arena, gamma_min=0.05, gamma_steps=10,
    )
    assert active is True
    refs = subgoal[None, :] + out
    for ref in refs:
        assert -1.0 + 0.18 <= ref[0] <= 1.0 - 0.18 + 1e-9
        assert -1.0 + 0.18 <= ref[1] <= 1.0 - 0.18 + 1e-9


def test_np_projection_gamma_floor():
    # Subgoal itself sits inside the obstacle → nothing feasible, return floor.
    offsets = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5]])
    subgoal = np.array([0.0, 0.0])
    obstacles = [(np.array([0.0, 0.0]), 1.0)]  # huge obstacle over subgoal
    out, gamma, active = project_affine_offsets_np(
        offsets, subgoal, obstacles,
        vehicle_radius=0.18, d_safe=0.15, gamma_min=0.2, gamma_steps=6,
    )
    assert active is True
    assert gamma == pytest.approx(0.2)


# ---------------------------------------------------------------------------
#  torch differentiable layer
# ---------------------------------------------------------------------------
def test_perpendicular_extent_value():
    # Offsets along x, path along x → perpendicular extent should be ~0.
    offsets = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])  # (1, 2, 2)
    path_dir = torch.tensor([[1.0, 0.0]])
    e = perpendicular_extent(offsets, path_dir)
    assert e.item() == pytest.approx(0.0, abs=1e-6)

    # Offsets along y, path along x → perpendicular extent = 1.0.
    offsets = torch.tensor([[[0.0, 1.0], [0.0, -0.7]]])
    e = perpendicular_extent(offsets, path_dir)
    assert e.item() == pytest.approx(1.0, abs=1e-6)


def test_torch_layer_identity_when_disabled(cfg):
    layer = FeasibilityProjectionLayer(enable_projection=False)
    z = torch.zeros(4, 6)
    offs = torch.tensor(np.array([[0.4, 0.0], [-0.2, 0.35], [-0.2, -0.35]]),
                        dtype=torch.float32)
    path_dir = torch.tensor([[1.0, 0.0]]).expand(4, 2)
    out, gamma = layer(z, offs, path_dir, w_allow=0.01, cfg=cfg)
    assert torch.allclose(gamma, torch.ones(4))


def test_torch_layer_limits_perpendicular_width(cfg):
    layer = FeasibilityProjectionLayer(enable_projection=True)
    # Wide formation offsets; force a tight width budget.
    offs = torch.tensor(np.array([[0.0, 0.6], [0.0, -0.6], [0.5, 0.0]]),
                        dtype=torch.float32)
    z = torch.zeros(1, 6)  # identity-ish decode (s = midpoint of [s_min, s_max])
    path_dir = torch.tensor([[1.0, 0.0]])
    w_allow = 0.2
    out, gamma = layer(z, offs, path_dir, w_allow=w_allow, cfg=cfg)
    e_after = perpendicular_extent(out, path_dir).item()
    assert e_after <= w_allow + 1e-5
    assert gamma.item() < 1.0


def test_torch_layer_gradient_flows(cfg):
    layer = FeasibilityProjectionLayer(enable_projection=True)
    z = torch.zeros(2, 6, requires_grad=True)
    offs = torch.tensor(np.array([[0.0, 0.6], [0.0, -0.6], [0.4, 0.0]]),
                        dtype=torch.float32)
    path_dir = torch.tensor([[1.0, 0.0]]).expand(2, 2)
    out, gamma = layer(z, offs, path_dir, w_allow=0.2, cfg=cfg)
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_torch_np_agree_on_effective_scale_direction(cfg):
    # Sanity: decoding then torch-projecting with a very wide budget matches
    # the raw decode (both leave the formation untouched).
    z_np = np.zeros(6)
    offs_np = decode_affine_action_np(z_np, np.array([[0.4, 0.0], [-0.2, 0.35],
                                                      [-0.2, -0.35]]), cfg)
    layer = FeasibilityProjectionLayer(enable_projection=True)
    z_t = torch.zeros(1, 6)
    base = torch.tensor(np.array([[0.4, 0.0], [-0.2, 0.35], [-0.2, -0.35]]),
                        dtype=torch.float32)
    path_dir = torch.tensor([[1.0, 0.0]])
    out, gamma = layer(z_t, base, path_dir, w_allow=100.0, cfg=cfg)
    assert gamma.item() == pytest.approx(1.0)
