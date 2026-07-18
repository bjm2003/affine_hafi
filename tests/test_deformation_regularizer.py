"""Unit tests for the clearance-modulated deformation regularizer.

The learned affine policy exhibited DEGENERATE SATURATION: it pinned anisotropy
(|s_x - s_y|) and rotation at the action-space corner in EVERY scenario, including
open space where deformation is physically meaningless (affine_usage_probe.py).
Root cause: the extra affine DOF were unpenalized, so they collapsed to an extreme.

This regularizer penalizes deforming AWAY from isotropy, gated by how much free
space surrounds the formation: full penalty in open space, ~0 in a tight passage.
These tests pin the scenario-correct behavior we need before spending a retrain on it:

    * isotropic action        → 0 penalty at any clearance
    * deform in OPEN (large clearance)  → full penalty
    * deform in TIGHT (small clearance) → ~0 penalty (deformation allowed)
    * default weight 0        → compute_reward emits no deform_reg term (no-op)
"""
from __future__ import annotations

import numpy as np
import pytest

from config import Config
from envs.rewards import compute_reward, deformation_penalty


# Max-deformation action: s_x=s_max (z3=+1), s_y=s_min (z4=-1), full rot + shear.
# Normalized aniso=shear=rot=1 → deform = 1^2 + 1^2 + 1^2 = 3.
Z_MAX_DEFORM = np.array([0.0, 0.0, 1.0, 1.0, -1.0, 1.0])
# Isotropic action: theta=0, s_x==s_y (z3==z4), kappa=0 → deform = 0.
Z_ISOTROPIC = np.array([0.0, 0.0, 0.0, 0.3, 0.3, 0.0])


def test_isotropic_action_never_penalized():
    cfg = Config()
    for clearance in (0.0, 0.5, cfg.deform_clearance_open, 5.0):
        assert deformation_penalty(Z_ISOTROPIC, clearance, cfg) == pytest.approx(0.0)


def test_open_space_full_penalty():
    cfg = Config()
    # clearance >= deform_clearance_open → gate = 1 → full deform magnitude (=3).
    pen = deformation_penalty(Z_MAX_DEFORM, cfg.deform_clearance_open + 0.5, cfg)
    assert pen == pytest.approx(3.0)


def test_tight_passage_no_penalty():
    cfg = Config()
    # clearance <= deform_clearance_tight → gate = 0 → deformation allowed for free.
    pen = deformation_penalty(Z_MAX_DEFORM, cfg.deform_clearance_tight - 0.1, cfg)
    assert pen == pytest.approx(0.0)


def test_gate_is_monotonic_and_linear_between_thresholds():
    cfg = Config()
    mid = 0.5 * (cfg.deform_clearance_tight + cfg.deform_clearance_open)
    pen_mid = deformation_penalty(Z_MAX_DEFORM, mid, cfg)
    # Midpoint clearance → gate = 0.5 → half of full deform (=1.5).
    assert pen_mid == pytest.approx(1.5)
    # Strictly increasing with clearance in the transition band.
    lo = deformation_penalty(Z_MAX_DEFORM, cfg.deform_clearance_tight + 0.05, cfg)
    hi = deformation_penalty(Z_MAX_DEFORM, cfg.deform_clearance_open - 0.05, cfg)
    assert lo < pen_mid < hi


def _reward_args(cfg):
    positions = np.array([[0.0, 0.5], [-0.4, -0.2], [0.4, -0.2]])
    nominal = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            nominal[i, j] = np.linalg.norm(positions[i] - positions[j])
    return dict(
        positions=positions,
        prev_center=np.array([0.0, 0.1]),
        goal=np.array([5.0, 0.0]),
        current_scale=1.0,
        prev_scale=1.0,
        nominal_dists=nominal,
        collision=False,
        cfg=cfg,
        scenario_name="open",
    )


def test_compute_reward_default_weight_is_noop():
    cfg = Config()
    assert cfg.w_deform_reg == 0.0
    _, comps = compute_reward(
        affine_action=Z_MAX_DEFORM, clearance=2.0, **_reward_args(cfg)
    )
    assert "deform_reg" not in comps


def test_compute_reward_penalizes_open_deformation():
    cfg = Config()
    cfg.w_deform_reg = 2.0
    _, comps = compute_reward(
        affine_action=Z_MAX_DEFORM, clearance=2.0, **_reward_args(cfg)
    )
    # gate=1, deform=3, weight=2 → component = -6.
    assert comps["deform_reg"] == pytest.approx(-6.0)


def test_compute_reward_allows_tight_deformation():
    cfg = Config()
    cfg.w_deform_reg = 2.0
    _, comps = compute_reward(
        affine_action=Z_MAX_DEFORM, clearance=0.1, **_reward_args(cfg)
    )
    assert comps["deform_reg"] == pytest.approx(0.0)


def test_compute_reward_skips_when_no_action():
    cfg = Config()
    cfg.w_deform_reg = 2.0
    # hafi_3d path passes affine_action=None → regularizer must not fire.
    _, comps = compute_reward(affine_action=None, clearance=2.0, **_reward_args(cfg))
    assert "deform_reg" not in comps
