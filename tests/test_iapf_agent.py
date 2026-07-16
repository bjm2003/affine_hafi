"""Unit tests for the classical IAPF high-level baseline (baselines.IAPFAgent).

Fast + MPC-free: IAPFAgent reads a duck-typed env (positions/goal/obstacles) and
emits a hafi_3d action (dx, dy, raw_scale). We assert its potential-field
invariants directly, independent of the low-level MPC.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines import IAPFAgent


class _Obs:
    def __init__(self, pos, radius, velocity=None):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.radius = float(radius)
        self.velocity = np.zeros(2) if velocity is None else np.asarray(velocity, dtype=np.float64)


class _FakeEnv:
    """Minimal stand-in exposing exactly what IAPFAgent reads."""

    def __init__(self, positions, goal, static_obs=None, dynamic_obs=None):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.static_obs = list(static_obs or [])
        self.dynamic_obs = list(dynamic_obs or [])


# Three vehicles whose centroid is the origin.
_TRI = [[0.0, 0.2], [-0.2, -0.1], [0.2, -0.1]]
_DUMMY_OBS = np.zeros(22, dtype=np.float32)   # predict ignores obs (privileged)


def test_heading_points_at_goal_when_clear():
    env = _FakeEnv(_TRI, goal=[2.0, 0.0])
    a, _ = IAPFAgent(env=env).predict(_DUMMY_OBS)
    dx, dy, raw_scale = float(a[0]), float(a[1]), float(a[2])
    assert dx > 0.95           # essentially straight toward the goal on +x
    assert abs(dy) < 0.05
    assert raw_scale == pytest.approx(0.0)   # no obstacles ⇒ nominal scale


def test_heading_deflects_away_from_offset_obstacle():
    # Obstacle offset to +y of the straight path ⇒ heading tilts toward -y
    # (steer away), while still making net forward progress.
    obs = [_Obs([0.8, 0.35], 0.3)]
    env = _FakeEnv(_TRI, goal=[2.0, 0.0], static_obs=obs)
    a, _ = IAPFAgent(env=env).predict(_DUMMY_OBS)
    dx, dy = float(a[0]), float(a[1])
    assert dy < -0.05          # steered away from the obstacle's (+y) side
    assert dx > 0.0            # still net forward toward the goal


def test_symmetric_obstacle_ahead_keeps_forward_heading():
    # Obstacle dead-ahead: bounded+capped repulsion never reverses the heading;
    # the team drives in and the MPC (hard-safety layer) splits around it.
    obs = [_Obs([0.8, 0.0], 0.4)]
    env = _FakeEnv(_TRI, goal=[2.0, 0.0], static_obs=obs)
    a, _ = IAPFAgent(env=env).predict(_DUMMY_OBS)
    assert float(a[0]) > 0.0   # forward component preserved (no entrance barrier)


def test_scale_shrinks_near_tight_obstacle():
    # Obstacle close to a vehicle (gap < d_tight) ⇒ full shrink command.
    obs = [_Obs([0.45, -0.1], 0.2)]   # ~0.05 m from vehicle (0.2,-0.1)
    env = _FakeEnv(_TRI, goal=[3.0, 0.0], static_obs=obs)
    a, _ = IAPFAgent(env=env).predict(_DUMMY_OBS)
    raw_scale = float(a[2])
    assert raw_scale < -0.5    # commanding a strong contraction


def test_distant_obstacle_does_not_perturb():
    # Obstacle well beyond both influence radius and scale band ⇒ no effect.
    obs = [_Obs([0.0, 5.0], 0.3)]
    env = _FakeEnv(_TRI, goal=[2.0, 0.0], static_obs=obs)
    a, _ = IAPFAgent(env=env).predict(_DUMMY_OBS)
    dx, dy, raw_scale = float(a[0]), float(a[1]), float(a[2])
    assert dx > 0.95
    assert abs(dy) < 0.05
    assert raw_scale == pytest.approx(0.0)


def test_unbound_env_raises():
    with pytest.raises(RuntimeError):
        IAPFAgent().predict(_DUMMY_OBS)


def test_action_shape_and_bounds():
    obs = [_Obs([0.8, 0.3], 0.3)]
    env = _FakeEnv(_TRI, goal=[1.5, 1.0], static_obs=obs)
    a, none = IAPFAgent(env=env).predict(_DUMMY_OBS)
    assert none is None
    assert a.shape == (3,)
    assert a.dtype == np.float32
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
