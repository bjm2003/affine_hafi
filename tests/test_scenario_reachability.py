"""Reachability invariants for every scenario generator.

Motivation: the first HAFI baseline run scored SR=3.3% at Gate G1 because goals
were physically unreachable — `sample_start_goal_pair` (open/dynamic) and
`u_trap` anchored the layout at the origin and pushed goals 6-10m away, landing
outside the +/-6m world where `world_clip` pins vehicles at the boundary.
`sequential_doorways` embedded the goal inside a 90-degree door wall.

These tests assert, for every registered scenario, that a sampled instance is
actually solvable in principle:
  1. start and goal lie inside the clipped arena
  2. start->goal separation fits the per-episode travel budget
  3. start and goal are clear of static obstacles (not spawned inside a wall)

If any of these fail, no policy can succeed there and training/eval numbers are
meaningless. Keep these strict.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import Config
from envs.scenario_generators import (
    build_scenario,
    TRAINING_SCENARIOS,
    KILLER_SCENARIOS,
)

ALL_SCENARIOS = TRAINING_SCENARIOS + KILLER_SCENARIOS
N_SAMPLES = 60


def _limits(cfg: Config):
    half = cfg.world_size / 2.0
    clip_hi = half - cfg.world_clip_margin
    budget = cfg.v_max * cfg.dt_rl * cfg.max_episode_steps  # single-axis straight-line
    clearance = cfg.vehicle_radius + cfg.d_safe             # center-to-obstacle min
    return clip_hi, budget, clearance


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_start_goal_in_bounds(name):
    cfg = Config()
    clip_hi, _, _ = _limits(cfg)
    for seed in range(N_SAMPLES):
        inst = build_scenario(name).sample(np.random.default_rng(seed), cfg)
        s = np.asarray(inst.start_center)
        g = np.asarray(inst.goal)
        assert (np.abs(s) <= clip_hi).all(), (
            f"{name} seed{seed}: start {s} outside clip bound +/-{clip_hi}"
        )
        assert (np.abs(g) <= clip_hi).all(), (
            f"{name} seed{seed}: goal {g} outside clip bound +/-{clip_hi} "
            f"(vehicles get pinned at the world boundary -> unreachable)"
        )


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_goal_reachable_within_travel_budget(name):
    cfg = Config()
    _, budget, _ = _limits(cfg)
    for seed in range(N_SAMPLES):
        inst = build_scenario(name).sample(np.random.default_rng(seed), cfg)
        sep = float(np.linalg.norm(np.asarray(inst.goal) - np.asarray(inst.start_center)))
        assert sep <= budget, (
            f"{name} seed{seed}: start-goal separation {sep:.2f}m exceeds "
            f"per-episode travel budget {budget:.2f}m (v_max*dt_rl*max_steps) -> timeout"
        )


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_start_and_goal_clear_of_obstacles(name):
    cfg = Config()
    _, _, clearance = _limits(cfg)
    for seed in range(N_SAMPLES):
        inst = build_scenario(name).sample(np.random.default_rng(seed), cfg)
        s = np.asarray(inst.start_center)
        g = np.asarray(inst.goal)
        for o in inst.static_obstacles:
            ds = float(np.linalg.norm(np.asarray(o.pos) - s))
            dg = float(np.linalg.norm(np.asarray(o.pos) - g))
            assert ds >= o.radius + clearance, (
                f"{name} seed{seed}: start {s} inside/too close to obstacle "
                f"at {o.pos} (r={o.radius:.2f}), dist={ds:.2f} < {o.radius + clearance:.2f}"
            )
            assert dg >= o.radius + clearance, (
                f"{name} seed{seed}: goal {g} inside/too close to obstacle "
                f"at {o.pos} (r={o.radius:.2f}), dist={dg:.2f} < {o.radius + clearance:.2f}"
            )
