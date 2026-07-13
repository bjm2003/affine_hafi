"""Smoke tests for scenario generators and FormationEnv.reset() / observation.

Run:
    pytest tests/test_scenarios_and_env.py -v
"""

from __future__ import annotations
import numpy as np
import pytest

from config import Config
from envs.formation_env import FormationEnv
from envs.scenario_generators import build_scenario, TRAINING_SCENARIOS, KILLER_SCENARIOS


@pytest.fixture
def cfg():
    return Config()


# ============================================================
#  Scenario generators
# ============================================================
@pytest.mark.parametrize("name", TRAINING_SCENARIOS + KILLER_SCENARIOS)
def test_scenario_sample(cfg, name):
    """Every registered scenario should produce a valid ScenarioInstance."""
    rng = np.random.default_rng(0)
    gen = build_scenario(name)
    inst = gen.sample(rng, cfg)

    assert inst.start_center.shape == (2,)
    assert inst.goal.shape == (2,)
    assert isinstance(inst.static_obstacles, list)
    assert isinstance(inst.dynamic_obstacles, list)
    assert isinstance(inst.metadata, dict)
    # start != goal
    assert np.linalg.norm(inst.start_center - inst.goal) > 0.1


def test_curved_slot_has_walls(cfg):
    rng = np.random.default_rng(0)
    gen = build_scenario("curved_slot")
    inst = gen.sample(rng, cfg)
    # Should have enough wall obstacles to actually form a corridor
    assert len(inst.static_obstacles) >= 20


def test_sequential_doorways_correct_count(cfg):
    rng = np.random.default_rng(0)
    gen = build_scenario("sequential_doorways", n_doorways=3)
    inst = gen.sample(rng, cfg)
    md = inst.metadata
    assert md["n_doorways"] == 3


def test_asymmetric_density(cfg):
    rng = np.random.default_rng(0)
    gen = build_scenario("asymmetric_density",
                         dense_side_n_static=10, sparse_side_n_static=2)
    inst = gen.sample(rng, cfg)
    md = inst.metadata
    assert md["dense_n"] == 10
    assert md["sparse_n"] == 2


def test_interior_injection_has_schedule(cfg):
    rng = np.random.default_rng(0)
    gen = build_scenario("interior_injection")
    inst = gen.sample(rng, cfg)
    assert len(inst.inject_schedule) == 1
    step, obs = inst.inject_schedule[0]
    assert step > 0
    assert obs.radius > 0


# ============================================================
#  FormationEnv basic API
# ============================================================
def test_env_reset_returns_valid_obs():
    env = FormationEnv(scenario_mode="open", seed=42)
    obs, info = env.reset()
    assert obs.shape == (22,)
    assert obs.dtype == np.float32
    assert (obs >= env.observation_space.low - 1e-6).all()
    assert (obs <= env.observation_space.high + 1e-6).all()
    assert "scenario" in info


def test_env_reset_positions_form_correct_shape():
    env = FormationEnv(scenario_mode="open", seed=42)
    env.reset()
    # 3 vehicles equilateral triangle, side ≈ d_form
    dists = np.array([
        np.linalg.norm(env.positions[i] - env.positions[j])
        for i in range(3) for j in range(i + 1, 3)
    ])
    assert np.allclose(dists, env.cfg.d_form, atol=1e-4), f"Formation not equilateral: {dists}"


def test_env_step_terminates_at_timeout():
    """Trivial: no MPC wiring yet, but env.step should still advance step count."""
    env = FormationEnv(scenario_mode="open", seed=42)
    env.reset()
    for _ in range(env.cfg.max_episode_steps + 1):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    # Since step doesn't move positions (TODO MPC), it never reaches goal → truncated
    assert truncated, "Env should truncate after max_episode_steps"


def test_env_action_spaces_by_type():
    env_a = FormationEnv(action_type="hafi_3d", scenario_mode="open")
    env_b = FormationEnv(action_type="affine_6d", scenario_mode="open")
    assert env_a.action_space.shape == (3,)
    assert env_b.action_space.shape == (6,)


def test_env_lidar_produces_finite_ranges():
    env = FormationEnv(scenario_mode="corridor", seed=42)
    obs, _ = env.reset()
    spatial = obs[:16]
    assert np.isfinite(spatial).all()
    assert (spatial >= 0).all() and (spatial <= 1).all()
    # In a corridor, expect at least some sectors to see walls (< 1.0)
    assert (spatial < 1.0).any(), "Corridor scenario should register walls in LiDAR"


def test_scenario_probs_normalize_and_pick():
    env = FormationEnv(scenario_mode="mixed", seed=0)
    picks = []
    for _ in range(200):
        env.reset()
        picks.append(env.current_scenario.metadata.get("scenario"))
    # Rough smoke check: at least 3 different scenarios sampled
    assert len(set(picks)) >= 3


# ============================================================
#  Formation offsets
# ============================================================
def test_formation_offsets_symmetry():
    from envs.formation_templates import build_formation_offsets
    for n in (2, 3, 4, 5, 7):
        off = build_formation_offsets(n, 0.8)
        assert off.shape == (n, 2)
        assert np.allclose(off.mean(axis=0), 0.0, atol=1e-6), \
            f"N={n} offsets don't sum to zero"
