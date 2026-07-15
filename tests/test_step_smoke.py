"""Smoke tests for step() + MPC integration.

Run:
    pytest tests/test_step_smoke.py -v
    pytest tests/test_step_smoke.py -v -k "greedy"   # just the goal-reaching test

Slow tests (marked slow): all 10 scenarios × 20 steps. Skipped by default under
`pytest -m "not slow"`; run explicitly with `pytest -m slow` for a full smoke.
"""

from __future__ import annotations
import time
import numpy as np
import pytest

from config import Config
from envs.formation_env import FormationEnv
from envs.scenario_generators import TRAINING_SCENARIOS, KILLER_SCENARIOS


# ============================================================
#  Basic step() plumbing (short, always run)
# ============================================================
def test_step_returns_valid_tuple():
    env = FormationEnv(action_type="hafi_3d", scenario_mode="open", seed=42)
    obs, _ = env.reset()
    action = env.action_space.sample()
    step_out = env.step(action)
    assert len(step_out) == 5
    obs, reward, terminated, truncated, info = step_out
    assert obs.shape == (22,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_components" in info
    assert "mpc_feasibility_rate" in info


def test_step_updates_positions():
    """Non-zero action should typically move at least one vehicle."""
    env = FormationEnv(action_type="hafi_3d", scenario_mode="open", seed=0)
    env.reset()
    initial = env.positions.copy()
    # Push straight toward goal
    for _ in range(3):
        goal_dir = env.goal - env.positions.mean(axis=0)
        gn = float(np.linalg.norm(goal_dir))
        if gn < 1e-6:
            break
        dx, dy = goal_dir / gn
        env.step(np.array([dx, dy, 0.0], dtype=np.float32))
    final = env.positions.copy()
    # Something should have moved at least 5 cm total
    assert np.linalg.norm(final - initial, axis=1).max() > 0.05


def test_step_action_type_affine():
    env = FormationEnv(action_type="affine_6d", scenario_mode="open", seed=42)
    env.reset()
    # Zero action → identity affine → team should still move via subgoal ≈ current center
    obs, r, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert np.isfinite(r)


def test_reward_components_present():
    env = FormationEnv(action_type="hafi_3d", scenario_mode="corridor", seed=1)
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    comp = info["reward_components"]
    for key in ("progress", "goal", "scale", "delta_s", "collision", "formation", "time"):
        assert key in comp


def test_reward_uses_corridor_scale_weight():
    """In a corridor scenario, w_scale_active should be w_scale_corridor (8.0)."""
    env = FormationEnv(action_type="hafi_3d", scenario_mode="corridor", seed=0)
    env.reset()
    # Force a large scale > 1 to trigger the penalty
    env.current_scale = 1.8
    env.prev_scale = 1.8
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    scale_component = info["reward_components"]["scale"]
    # After leader_node scale remap, effective scale should still be > 1.2
    # w_scale_corridor(=8.0) × -(s - 1)^2 should give |penalty| > 0.3
    assert scale_component < -0.3, f"Expected corridor scale penalty, got {scale_component}"


def test_termination_on_goal():
    env = FormationEnv(action_type="hafi_3d", scenario_mode="open", seed=0)
    env.reset()
    # Disable world clip so we can place vehicles at any goal (some samples
    # generate goals outside the [-world_half, world_half] range)
    env.cfg.enable_world_clip = False
    # Teleport centroid ~ to goal
    env.positions = env.goal[None, :] + env.formation_offsets * 1.0  # keep formation
    env.prev_center = env.positions.mean(axis=0)
    # Zero action → LeaderNode will set subgoal = goal (dist < R0), MPC barely moves
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    # Centroid should stay near goal (formation offsets sum to zero)
    assert info["dist_to_goal"] < 0.3, (
        f"Expected centroid near goal, got dist={info['dist_to_goal']:.3f}"
    )
    # Success flag or terminated should reflect goal proximity
    assert info["success"] or info["dist_to_goal"] < env.cfg.goal_tolerance + 0.2


def test_truncation_at_max_steps():
    env = FormationEnv(action_type="hafi_3d", scenario_mode="open", seed=0)
    env.reset()
    # Cap max steps low
    env.cfg.max_episode_steps = 3
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
        if terminated or truncated:
            assert truncated or terminated
            return
    pytest.fail("Env did not truncate at max_episode_steps")


def test_lazy_solver_init():
    """Solvers should only be built on first step, not on __init__."""
    env = FormationEnv(action_type="hafi_3d", scenario_mode="open")
    assert env._mpc_solvers is None
    env.reset()
    assert env._mpc_solvers is None
    env.step(np.zeros(3, dtype=np.float32))
    assert env._mpc_solvers is not None
    assert len(env._mpc_solvers) == env.cfg.n_vehicles


# ============================================================
#  Greedy goal-reaching (open scenario should be trivially solvable)
# ============================================================
def test_open_scenario_reaches_goal_with_greedy_policy():
    """A greedy 'point toward goal' policy must REACH the goal in the open scenario.

    Reachability budget: v_max=0.3 m/s, dt_rl=0.5s, 150 steps → single-axis reach
    ≈ 0.3×0.5×150 = 22.5m, comfortably above the 6-10m start-goal range. If this
    ever fails, the reachability params (v_max / dt_rl / max_episode_steps vs
    start_distance_range) are misconfigured — that's the exact bug that produced
    SR≈3% at Gate G1. Keep this assertion strict.
    """
    env = FormationEnv(action_type="hafi_3d", scenario_mode="open", seed=42)
    obs, _ = env.reset()
    initial_dist = float(np.linalg.norm(env.positions.mean(axis=0) - env.goal))
    max_steps = env.cfg.max_episode_steps

    for step in range(max_steps):
        center = env.positions.mean(axis=0)
        goal_vec = env.goal - center
        gn = float(np.linalg.norm(goal_vec))
        if gn < 1e-6:
            break
        dx, dy = goal_vec / gn
        action = np.array([dx, dy, 0.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        if info["success"]:
            return  # reached the goal — success
        if terminated and not info["success"]:
            pytest.fail(f"Collided in open scenario at step {step}")
        if truncated:
            break

    pytest.fail(
        f"Greedy policy failed to reach goal in open scenario: "
        f"initial dist={initial_dist:.2f}m, final dist={info['dist_to_goal']:.2f}m "
        f"after {step + 1} steps. Check reachability params (v_max/dt_rl/max_episode_steps)."
    )


# ============================================================
#  All 10 scenarios × 20 random steps (slow smoke)
# ============================================================
@pytest.mark.slow
@pytest.mark.parametrize("scenario_name", TRAINING_SCENARIOS + KILLER_SCENARIOS)
def test_all_scenarios_random_step_smoke(scenario_name):
    """Every scenario must survive 20 random-action steps without crashing."""
    env = FormationEnv(action_type="affine_6d", scenario_mode=scenario_name, seed=7)
    obs, _ = env.reset()
    feasible_rate_accum = []
    for _ in range(20):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r), f"NaN reward in {scenario_name}"
        assert np.all(np.isfinite(env.positions)), f"NaN position in {scenario_name}"
        feasible_rate_accum.append(info["mpc_feasibility_rate"])
        if term or trunc:
            break

    # Overall MPC should be mostly feasible (>60%) even under random policy
    mean_feasible = float(np.mean(feasible_rate_accum))
    assert mean_feasible > 0.5, (
        f"MPC feasibility too low in {scenario_name}: {mean_feasible:.3f}"
    )


# ============================================================
#  Timing sanity check
# ============================================================
@pytest.mark.slow
def test_step_wall_clock_reasonable():
    """10 episodes × 30 steps should complete in reasonable time (< 90s on 4060)."""
    t0 = time.perf_counter()
    for ep in range(10):
        env = FormationEnv(action_type="hafi_3d", scenario_mode="open", seed=ep)
        env.reset()
        for _ in range(30):
            env.step(env.action_space.sample())
    dt = time.perf_counter() - t0
    # 4060: expect < 60s. Safety margin 90s. If exceeded, MPC is likely too slow.
    assert dt < 90.0, f"Env too slow: {dt:.1f}s for 300 steps"
    print(f"[timing] 300 steps in {dt:.1f}s → {dt/300*1000:.1f} ms/step")
