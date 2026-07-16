"""Unit tests for the classical geometric-affine baseline (GeometricAffineAgent).

Fast + MPC-free: the agent reads a duck-typed env (positions/goal/obstacles/
formation_offsets) and emits an affine_6d action (dx, dy, theta, s_x, s_y, kappa).
We assert its geometric invariants directly, independent of the low-level MPC and
the C2 projection (which run downstream in the real env):

    * clear path  → heading at goal, formation nominal (theta/s_x/s_y/kappa ≈ 0)
    * travel axis → depth axis aligned, heading angle wrapped into [-90°, 90°]
    * tight gap   → anisotropic cross-path shrink (s_y < nominal, s_x stays 1.0)
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines import GeometricAffineAgent
from config import Config
from envs.formation_templates import build_formation_offsets


class _Obs:
    def __init__(self, pos, radius, velocity=None):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.radius = float(radius)
        self.velocity = np.zeros(2) if velocity is None else np.asarray(velocity, dtype=np.float64)


class _FakeEnv:
    """Minimal stand-in exposing exactly what GeometricAffineAgent reads."""

    def __init__(self, positions, goal, static_obs=None, dynamic_obs=None, cfg=None):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.static_obs = list(static_obs or [])
        self.dynamic_obs = list(dynamic_obs or [])
        self.cfg = cfg or Config()
        self.formation_offsets = build_formation_offsets(
            self.cfg.n_vehicles, self.cfg.d_form,
        )


# Three vehicles whose centroid is the origin.
_TRI = [[0.0, 0.2], [-0.2, -0.1], [0.2, -0.1]]
_DUMMY_OBS = np.zeros(22, dtype=np.float32)   # predict ignores obs (privileged)
_CFG = Config()


def _agent(env):
    return GeometricAffineAgent(cfg=_CFG, env=env)


def test_clear_path_is_nominal_formation():
    env = _FakeEnv(_TRI, goal=[2.0, 0.0])
    a, _ = _agent(env).predict(_DUMMY_OBS)
    dx, dy, theta, sx, sy, kappa = (float(v) for v in a)
    assert dx > 0.95 and abs(dy) < 0.05         # heading straight at goal
    assert theta == pytest.approx(0.0, abs=1e-4)  # depth axis already along +x
    assert sx == pytest.approx(0.0)             # s_x nominal
    assert sy == pytest.approx(0.0)             # s_y nominal (clear ahead)
    assert kappa == pytest.approx(0.0)          # no shear


def test_theta_wraps_travel_axis_into_pm_90deg():
    # Goal up-left ⇒ heading ≈ 135°. A corridor is an undirected line, so the
    # depth axis aligns to the *axis*: 135° wraps to -45° ⇒ z_theta ≈ -0.5.
    env = _FakeEnv(_TRI, goal=[-1.0, 1.0])
    a, _ = _agent(env).predict(_DUMMY_OBS)
    dx, dy, theta = float(a[0]), float(a[1]), float(a[2])
    assert dx < 0.0 and dy > 0.0                # heading unchanged (up-left)
    assert theta == pytest.approx(-0.5, abs=0.05)


def test_tight_gap_shrinks_sy_only():
    # A symmetric perpendicular gap ahead ⇒ anisotropic cross-path contraction:
    # s_y drops below nominal while s_x stays at 1.0 (the move isotropic can't do).
    obs = [_Obs([1.0, 0.55], 0.1), _Obs([1.0, -0.55], 0.1)]
    env = _FakeEnv(_TRI, goal=[3.0, 0.0], static_obs=obs)
    a, _ = _agent(env).predict(_DUMMY_OBS)
    dx, sx, sy = float(a[0]), float(a[3]), float(a[4])
    assert dx > 0.9                             # symmetric walls ⇒ still forward
    assert sy < -0.05                           # cross-path base squeezed
    assert sx == pytest.approx(0.0)             # along-path spacing preserved


def test_sy_never_grows_past_nominal():
    # A wide gap must not inflate the formation: s_y is capped at nominal.
    obs = [_Obs([1.0, 2.0], 0.1), _Obs([1.0, -2.0], 0.1)]
    env = _FakeEnv(_TRI, goal=[3.0, 0.0], static_obs=obs)
    a, _ = _agent(env).predict(_DUMMY_OBS)
    assert float(a[4]) <= 1e-6                  # z_sy ≤ 0 ⇒ s_y ≤ 1.0


def test_distant_obstacle_does_not_perturb():
    # Obstacle outside the look-ahead band ⇒ nominal formation.
    obs = [_Obs([0.0, 5.0], 0.3)]
    env = _FakeEnv(_TRI, goal=[2.0, 0.0], static_obs=obs)
    a, _ = _agent(env).predict(_DUMMY_OBS)
    assert abs(float(a[2])) < 1e-4 and float(a[3]) == pytest.approx(0.0)
    assert float(a[4]) == pytest.approx(0.0) and float(a[5]) == pytest.approx(0.0)


def test_unbound_env_raises():
    with pytest.raises(RuntimeError):
        GeometricAffineAgent().predict(_DUMMY_OBS)


def test_action_shape_and_bounds():
    obs = [_Obs([1.0, 0.5], 0.2)]
    env = _FakeEnv(_TRI, goal=[1.5, 1.0], static_obs=obs)
    a, none = _agent(env).predict(_DUMMY_OBS)
    assert none is None
    assert a.shape == (6,)
    assert a.dtype == np.float32
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
