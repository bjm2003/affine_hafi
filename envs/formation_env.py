"""
FormationEnv — Gym Env for multi-robot formation navigation.

Self-built from HAFI spec (config.py). Supports both HAFI baseline (3D action)
and Affine method (6D action).

Observation (22-dim, aligned with HAFI):
    - 16-sector LiDAR (from leader car, min-pooled per sector, normalized [0,1])
    - 6-dim team state: [gx, gy, vx_center, vy_center, current_scale, prev_scale]

Action:
    - hafi_3d:   (dx, dy, scale) ∈ [-1, 1]^3
    - affine_6d: (dx, dy, theta, s_x, s_y, kappa) ∈ [-1, 1]^6

Termination:
    - Success: ‖p_c - g‖ ≤ goal_tolerance
    - Collision: any robot-obstacle or robot-robot dist < r_v + r_m + d_safe
    - Timeout: t ≥ max_episode_steps

STATUS: reset() + observation are wired up. step() delegates to MPC (TODO).
        Reward computation (envs/rewards.py) is a stub for now.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from config import Config
from envs.formation_templates import build_formation_offsets, nominal_pairwise_distances
from envs.scenario_generators import build_scenario, TRAINING_SCENARIOS, ScenarioInstance
from envs.lidar_sim import simulate_lidar, derotate_lidar, lidar_to_spatial_dirs


class FormationEnv(gym.Env):
    """Multi-robot formation navigation Env (HAFI-compatible spec)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        cfg: Optional[Config] = None,
        action_type: str = "affine_6d",
        scenario_mode: str = "mixed",       # "mixed" (curriculum) or scenario name
        scenario_probs: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg or Config()
        self.action_type = action_type
        self.scenario_mode = scenario_mode
        self.scenario_probs = scenario_probs or {
            n: p for n, p in zip(TRAINING_SCENARIOS, self.cfg.train_scenario_probs)
        }
        self.render_mode = render_mode

        # RNG (Gym reset() will re-seed)
        self._rng = np.random.default_rng(seed)

        # ============ Observation space (22-dim, matches HAFI) ============
        obs_dim = self.cfg.n_spatial_dirs + self.cfg.n_self_features
        assert obs_dim == 22, f"Expected 22-dim obs, got {obs_dim}"
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0] * self.cfg.n_spatial_dirs
                + [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0] * self.cfg.n_spatial_dirs
                + [1.0, 1.0, 1.0, 1.0, self.cfg.s_max, self.cfg.s_max],
                dtype=np.float32,
            ),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ============ Action space ============
        if action_type == "hafi_3d":
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        elif action_type == "affine_6d":
            self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        # ============ Formation geometry ============
        self.formation_offsets = build_formation_offsets(
            self.cfg.n_vehicles, self.cfg.d_form,
        )
        self.nominal_dists = nominal_pairwise_distances(self.formation_offsets)

        # ============ Dynamic state (reset() populates) ============
        self.positions: np.ndarray = np.zeros((self.cfg.n_vehicles, 2))
        self.velocities: np.ndarray = np.zeros((self.cfg.n_vehicles, 2))
        self.yaw: float = 0.0                       # team heading (leader / centroid yaw)
        self.goal: np.ndarray = np.zeros(2)
        self.current_scenario: Optional[ScenarioInstance] = None
        self.static_obs: list = []
        self.dynamic_obs: list = []
        self.pending_injections: list = []          # [(step, ObstacleDynamic), ...]
        self.step_count: int = 0
        self.current_scale: float = 1.0
        self.prev_scale: float = 1.0
        self.prev_center: np.ndarray = np.zeros(2)

        # MPC solver (created lazily to defer JIT build)
        self._mpc_solver = None

    # ================================================================
    #  Gym API
    # ================================================================
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # 1. Pick scenario
        scenario_name = self._pick_scenario(options)
        scen_kwargs = (options or {}).get("scenario_kwargs", {})
        scenario_gen = build_scenario(scenario_name, **scen_kwargs)
        instance: ScenarioInstance = scenario_gen.sample(self._rng, self.cfg)

        # 2. Initialize positions from formation offset around start_center
        self.current_scenario = instance
        self.yaw = instance.start_orientation
        self.goal = instance.goal.copy()
        self.static_obs = list(instance.static_obstacles)
        self.dynamic_obs = list(instance.dynamic_obstacles)
        self.pending_injections = list(instance.inject_schedule)

        R = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw),  np.cos(self.yaw)],
        ])
        rotated_offsets = self.formation_offsets @ R.T
        self.positions = instance.start_center[None, :] + rotated_offsets

        # 3. Zero velocities
        self.velocities = np.zeros((self.cfg.n_vehicles, 2))

        # 4. Reset counters + scale state
        self.step_count = 0
        self.current_scale = 1.0
        self.prev_scale = 1.0
        self.prev_center = self.positions.mean(axis=0)

        # 5. Reset MPC warm start (if solver exists)
        if self._mpc_solver is not None:
            self._mpc_solver.reset_warm_start()

        obs = self._get_observation()
        info = {
            "scenario": instance.metadata,
            "n_static": len(self.static_obs),
            "n_dynamic": len(self.dynamic_obs),
            "n_pending_inject": len(self.pending_injections),
        }
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # ============ 1. Decode high-level action to team-level intention ============
        # TODO: hafi_3d: (dx, dy, scale) → subgoal + scale
        # TODO: affine_6d: (dx, dy, theta, s_x, s_y, kappa) → affine transformation A(z)
        #        → per-vehicle references via A(z) · D_0 + p_c_ref

        # ============ 2. Low-level MPC per vehicle ============
        # TODO: For each vehicle i, solve MPC to track affine-transformed reference
        # TODO: Execute mpc_steps_per_rl MPC steps (dt_rl / dt_mpc = 5 by default)

        # ============ 3. Propagate dynamic obstacles + inject scheduled ============
        self._propagate_dynamic_obstacles()
        self._process_injections()

        # ============ 4. Reward ============
        # TODO: import from envs.rewards
        reward = 0.0

        # ============ 5. Termination check ============
        center = self.positions.mean(axis=0)
        dist_to_goal = float(np.linalg.norm(center - self.goal))
        success = dist_to_goal < self.cfg.goal_tolerance
        collision = self._check_collision()
        terminated = success or collision
        truncated = self.step_count >= self.cfg.max_episode_steps

        # ============ 6. Bookkeeping ============
        self.prev_center = center.copy()
        self.prev_scale = self.current_scale
        self.step_count += 1

        obs = self._get_observation()
        info = {
            "success": success,
            "collision": collision,
            "dist_to_goal": dist_to_goal,
            "center": center,
            "current_scale": self.current_scale,
        }
        return obs, reward, terminated, truncated, info

    # ================================================================
    #  Observation construction (mirrors deploy/commander_node.py)
    # ================================================================
    def _get_observation(self) -> np.ndarray:
        leader_idx = 0
        leader_pos = self.positions[leader_idx]
        teammates = np.delete(self.positions, leader_idx, axis=0)

        # 1. Simulate LiDAR from leader
        lidar_ranges_body = simulate_lidar(
            origin=leader_pos,
            yaw=self.yaw,
            static_obs=self.static_obs,
            dynamic_obs=self.dynamic_obs,
            n_rays=self.cfg.n_lidar_rays,
            max_dist=self.cfg.lidar_max_dist,
            teammate_positions=teammates,
            teammate_radius=self.cfg.vehicle_radius,
        )
        # 2. Derotate to world frame
        lidar_world = derotate_lidar(lidar_ranges_body, self.yaw)

        # 3. Filter teammate rays (redundant with teammate_positions above, but
        #    keeps parity with deploy pipeline that filters post-hoc)
        # (already handled in simulate_lidar by adding teammates)

        # 4. Sector pooling → 16-dim spatial obs
        spatial = lidar_to_spatial_dirs(
            lidar_world, self.cfg.n_spatial_dirs, self.cfg.lidar_max_dist,
        )
        if self.cfg.no_spatial_obs:
            spatial = np.ones(self.cfg.n_spatial_dirs, dtype=np.float32)

        # 5. 6-dim team state
        center = self.positions.mean(axis=0)
        goal_vec = self.goal - center
        rho_max = self.cfg.world_size * np.sqrt(2)
        gx = float(np.clip(goal_vec[0] / rho_max, -1.0, 1.0))
        gy = float(np.clip(goal_vec[1] / rho_max, -1.0, 1.0))

        if self.step_count == 0:
            vx = vy = 0.0
        else:
            velocity = (center - self.prev_center) / self.cfg.dt_rl
            vx = float(np.clip(velocity[0] / self.cfg.v_max, -1.0, 1.0))
            vy = float(np.clip(velocity[1] / self.cfg.v_max, -1.0, 1.0))

        self_obs = np.array(
            [gx, gy, vx, vy, self.current_scale, self.prev_scale],
            dtype=np.float32,
        )

        return np.concatenate([spatial, self_obs]).astype(np.float32)

    # ================================================================
    #  Internal helpers
    # ================================================================
    def _pick_scenario(self, options: Optional[Dict[str, Any]]) -> str:
        """Sample scenario name based on mode + probs."""
        if options and "scenario" in options:
            return options["scenario"]

        if self.scenario_mode == "mixed":
            names = list(self.scenario_probs.keys())
            probs = np.array([self.scenario_probs[n] for n in names], dtype=np.float64)
            probs /= probs.sum()
            return str(self._rng.choice(names, p=probs))
        return self.scenario_mode

    def _propagate_dynamic_obstacles(self) -> None:
        """Advance dynamic obstacle positions by dt_rl."""
        dt = self.cfg.dt_rl
        world_half = self.cfg.world_size / 2.0
        for o in self.dynamic_obs:
            o.pos = o.pos + dt * o.velocity
            # Bounce off world bounds (soft)
            if abs(o.pos[0]) > world_half:
                o.velocity[0] *= -1
                o.pos[0] = np.clip(o.pos[0], -world_half, world_half)
            if abs(o.pos[1]) > world_half:
                o.velocity[1] *= -1
                o.pos[1] = np.clip(o.pos[1], -world_half, world_half)

    def _process_injections(self) -> None:
        """Add scheduled dynamic obstacles when their inject_step is reached."""
        due = [(s, o) for s, o in self.pending_injections if s <= self.step_count]
        for _, obs in due:
            self.dynamic_obs.append(obs)
        self.pending_injections = [(s, o) for s, o in self.pending_injections if s > self.step_count]

    def _check_collision(self) -> bool:
        """Return True if any robot-obs or robot-robot pair violates safety margin."""
        r_v = self.cfg.vehicle_radius
        d_safe = self.cfg.d_safe

        # Robot-obstacle
        for i in range(self.cfg.n_vehicles):
            for o in self.static_obs:
                if np.linalg.norm(self.positions[i] - o.pos) < r_v + o.radius + d_safe * 0.0:
                    # Use d_safe=0 for HARD collision (dist_safe is the reward-shaping margin)
                    if np.linalg.norm(self.positions[i] - o.pos) < r_v + o.radius:
                        return True
            for o in self.dynamic_obs:
                if np.linalg.norm(self.positions[i] - o.pos) < r_v + o.radius:
                    return True

        # Robot-robot
        for i in range(self.cfg.n_vehicles):
            for j in range(i + 1, self.cfg.n_vehicles):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < 2 * r_v:
                    return True
        return False

    def render(self):
        # TODO: matplotlib visualization for debug + eval videos
        pass

    def close(self):
        pass
