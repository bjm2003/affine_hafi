"""Classical (non-learned) high-level baselines.

These controllers emit the *same* action as the learned HAFI high-level and run
over the *same* per-vehicle MPC, so the comparison isolates exactly one thing:
the intention layer (hand-crafted vs learned). Everything downstream is shared.

Information model — deliberately *privileged*:
    A classical potential field is defined over a known obstacle map, so we give
    these agents the ground-truth obstacle geometry + goal (via the env), rather
    than the 22-dim LiDAR observation. This is both the textbook formulation and
    the *stronger* baseline: the leader's on-board LiDAR (lidar_max_dist=1.5 m,
    min-pooled) also returns its own teammates at ~0.4–0.8 m, which a naive APF
    would treat as obstacles. Feeding the classical controller the true obstacle
    map removes that self-inflicted handicap, so "learned-from-LiDAR beats
    classical-with-full-map" is a claim we can defend.

Because the eval runner only calls ``model.predict(obs, ...)``, these agents read
the live env state directly; bind the env before rolling out (constructor arg or
``bind_env``). The env is mutated in place each step, so reads at predict time
reflect the current true state.

Agent interface (duck-typed, matches SB3):
    predict(obs, deterministic=True) -> (action: np.ndarray, None)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _read_privileged_scene(env):
    """Pull the ground-truth scene the classical controllers act on.

    Returns (positions (N,2), centroid (2,), goal (2,), obstacles) where
    obstacles is a list of ``(pos (2,), radius)`` from static + dynamic obstacles.
    """
    positions = np.asarray(env.positions, dtype=np.float64)          # (N, 2)
    centroid = positions.mean(axis=0)
    goal = np.asarray(env.goal, dtype=np.float64).reshape(2)
    obstacles = [(np.asarray(o.pos, dtype=np.float64).reshape(2), float(o.radius))
                 for o in env.static_obs]
    obstacles += [(np.asarray(o.pos, dtype=np.float64).reshape(2), float(o.radius))
                  for o in env.dynamic_obs]
    return positions, centroid, goal, obstacles


def _apf_heading(
    centroid: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Tuple[np.ndarray, float]],
    *,
    k_att: float,
    k_rep: float,
    rho0: float,
    rep_cap: float,
    gnron: bool,
    gnron_r: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Bounded + capped attractive/repulsive potential-field heading.

    Shared by the classical baselines (IAPF and the geometric-affine rule). The
    repulsion is capped strictly below ``k_att`` so the normalized heading always
    retains a forward-to-goal component — the low-level MPC is the hard-safety
    layer, so this only needs to *steer* (never barrier the team out of a gap).

    Returns ``(heading_unit, att_unit, rep_vec, rep_norm, goal_dist)`` — enough
    for callers to add their own escape terms (e.g. IAPF's optional vortex).
    """
    to_goal = goal - centroid
    goal_dist = float(np.linalg.norm(to_goal))
    att = to_goal / goal_dist if goal_dist > 1e-9 else np.array([1.0, 0.0])

    rep = np.zeros(2, dtype=np.float64)
    for opos, orad in obstacles:
        diff = centroid - opos
        dd = float(np.linalg.norm(diff))
        if dd < 1e-9:
            continue
        d_surf = dd - orad                          # centroid → obstacle surface
        if d_surf < rho0:
            w = np.clip((rho0 - d_surf) / rho0, 0.0, 1.0)
            rep += (k_rep * w) * (diff / dd)        # push away from obstacle
    if gnron:
        rep *= min(goal_dist / gnron_r, 1.0)        # fade repulsion near the goal
    rep_norm = float(np.linalg.norm(rep))
    if rep_norm > rep_cap:
        rep *= rep_cap / rep_norm
        rep_norm = rep_cap

    force = k_att * att + rep
    fn = float(np.linalg.norm(force))
    heading = force / fn if fn > 1e-9 else att
    return heading, att, rep, rep_norm, goal_dist


class IAPFAgent:
    """Improved Artificial Potential Field high-level controller (hafi_3d).

    Emits a 3D ``hafi_3d`` action ``(dx, dy, raw_scale)`` in ``[-1, 1]^3``:

        att = normalize(goal − centroid)                     # pull toward goal
        rep = Σ_obs  w(d_surf) · normalize(centroid − obs)   # push off obstacles
        (dx, dy) = normalize(k_att · att + rep)              # blended heading
        raw_scale ∈ [-1, 0]  (shrink as the tightest obstacle gap narrows)

    computed from the ground-truth obstacle map (see module docstring).

    "Improved" vs. a vanilla APF:
        * GNRON modulation — repulsion fades near the goal so obstacles adjacent
          to the target can't block arrival (goals-non-reachable-near-obstacles).
        * Vortex escape — when attractive and repulsive forces nearly cancel (a
          local minimum), a tangential component slips the team around the
          obstacle instead of stalling head-on.

    The scale action shrinks the formation as the nearest obstacle surface
    approaches any vehicle, so the team squeezes through gaps — the isotropic
    analogue of HAFI's learned scale head — and never grows past nominal.
    """

    def __init__(
        self,
        cfg=None,
        env=None,
        k_att: float = 1.0,
        k_rep: float = 0.6,
        rho0: float = 1.2,          # repulsion influence radius (m, surface dist)
        rep_cap: float = 0.85,      # cap on |rep|; MUST be < k_att (see predict)
        gnron: bool = True,         # scale repulsion by distance-to-goal
        gnron_r: float = 0.8,       # goal-distance (m) at/above which rep is full
        vortex: bool = False,       # tangential nudge at local minima (off: the
                                    # rep cap already guarantees forward progress,
                                    # and a vortex shoves the team into corridor
                                    # walls; only useful for isolated obstacles)
        vortex_gain: float = 1.0,
        vortex_stall: float = 0.25,  # fire vortex only when |net force| below this
        d_clear: float = 0.80,      # nearest gap ≥ this (m) ⇒ nominal scale
        d_tight: float = 0.25,      # nearest gap ≤ this (m) ⇒ full shrink (-1)
    ):
        self.cfg = cfg
        self.env = env
        self.k_att = float(k_att)
        self.k_rep = float(k_rep)
        self.rho0 = float(rho0)
        self.rep_cap = float(rep_cap)
        self.gnron = bool(gnron)
        self.gnron_r = float(gnron_r)
        self.vortex = bool(vortex)
        self.vortex_gain = float(vortex_gain)
        self.vortex_stall = float(vortex_stall)
        self.d_clear = float(d_clear)
        self.d_tight = float(d_tight)

    def bind_env(self, env) -> "IAPFAgent":
        self.env = env
        return self

    # -- SB3-compatible inference API ------------------------------------
    def predict(
        self, obs, deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        env = self.env
        if env is None:
            raise RuntimeError("IAPFAgent needs an env bound (constructor or bind_env).")

        positions, centroid, goal, obstacles = _read_privileged_scene(env)

        # Bounded + capped potential-field heading (see _apf_heading docstring:
        # |rep| < k_att guarantees the normalized heading never reverses, so the
        # team keeps entering the gap while the MPC handles close-in avoidance).
        heading, att, rep, rep_norm, goal_dist = _apf_heading(
            centroid, goal, obstacles,
            k_att=self.k_att, k_rep=self.k_rep, rho0=self.rho0,
            rep_cap=self.rep_cap, gnron=self.gnron, gnron_r=self.gnron_r,
        )

        # --- optional vortex escape at genuine local minima (off by default) ---
        if self.vortex and rep_norm > 1e-6:
            force = self.k_att * att + rep
            if float(np.linalg.norm(force)) < self.vortex_stall:
                tangent = np.array([-rep[1], rep[0]]) / (rep_norm + 1e-9)
                if float(np.dot(tangent, att)) < 0.0:
                    tangent = -tangent
                force = force + self.vortex_gain * rep_norm * tangent
                fn = float(np.linalg.norm(force))
                heading = force / fn if fn > 1e-9 else att

        dx = float(np.clip(heading[0], -1.0, 1.0))
        dy = float(np.clip(heading[1], -1.0, 1.0))

        # --- scale action: shrink as the tightest obstacle gap narrows ---
        nearest_gap = np.inf
        for opos, orad in obstacles:
            d_veh = float(np.min(np.linalg.norm(positions - opos[None, :], axis=1)))
            nearest_gap = min(nearest_gap, d_veh - orad)
        if not np.isfinite(nearest_gap):
            raw_scale = 0.0                          # no obstacles → nominal
        else:
            span = max(self.d_clear - self.d_tight, 1e-6)
            frac = float(np.clip((nearest_gap - self.d_tight) / span, 0.0, 1.0))
            raw_scale = frac - 1.0                   # tight→-1 (shrink), clear→0

        action = np.array([dx, dy, raw_scale], dtype=np.float32)
        return action, None


class GeometricAffineAgent:
    """Hand-crafted geometric affine high-level controller (affine_6d).

    The classical *affine* counterpart to IAPF: instead of the learned policy's
    6D intention, it applies a fixed geometric rule that emits the same
    ``affine_6d`` action ``z = (dx, dy, theta, s_x, s_y, kappa) ∈ [-1, 1]^6`` and
    runs the *same* per-vehicle MPC + C2 feasibility projection. This isolates
    exactly one variable — learned affine intention vs hand-crafted affine
    geometry — the cleanest ablation for the paper's core claim.

    Rule (privileged obstacle map, see module docstring):
        heading  — bounded+capped APF (identical to IAPF) → (dx, dy) subgoal dir.
        theta    — rotate the formation so its depth axis (formation-x, the
                   leader→rear apex) aligns with the travel *axis*. Because the
                   action bound is ±theta_max = ±90° and a corridor is an
                   *undirected* line, the heading angle is wrapped into
                   [-90°, 90°]: this always represents the local passage axis, so
                   the base axis (formation-y) ends up perpendicular to travel.
        s_y      — anisotropic cross-path squeeze: shrink the base extent just
                   enough to fit the free width measured perpendicular to travel
                   at a look-ahead point (capped at nominal, only ever shrinks).
        s_x      — nominal (1.0): keep along-path spacing. This is exactly the
                   move isotropic scaling *cannot* make — narrow across without
                   crowding fore-aft — so it should clear tight gaps IAPF stalls
                   on, while still lacking the learned policy's adaptivity to
                   curved / asymmetric / dynamic gaps (κ shear, non-axis-aligned
                   rotation) that this fixed rule leaves on the table (κ = 0).

    Feasibility is guaranteed downstream by the same C2 projection the learned
    affine method uses (bind the agent to a projection-ON env), so the geometric
    intention only needs to be *reasonable*, not provably collision-free.
    """

    def __init__(
        self,
        cfg=None,
        env=None,
        k_att: float = 1.0,
        k_rep: float = 0.6,
        rho0: float = 1.2,
        rep_cap: float = 0.85,
        gnron: bool = True,
        gnron_r: float = 0.8,
        theta_max: float = np.pi / 2,   # must match env.affine_theta_max
        lookahead: float = 1.0,         # (m) ahead of centroid to probe the gap
        along_window: float = 1.2,      # (m) fore-aft band of obstacles to scan
        fit_margin: float = 0.05,       # (m) extra clearance folded into s_y fit
    ):
        self.cfg = cfg
        self.env = env
        self.k_att = float(k_att)
        self.k_rep = float(k_rep)
        self.rho0 = float(rho0)
        self.rep_cap = float(rep_cap)
        self.gnron = bool(gnron)
        self.gnron_r = float(gnron_r)
        self.theta_max = float(theta_max)
        self.lookahead = float(lookahead)
        self.along_window = float(along_window)
        self.fit_margin = float(fit_margin)

    def bind_env(self, env) -> "GeometricAffineAgent":
        self.env = env
        return self

    # -- SB3-compatible inference API ------------------------------------
    def predict(
        self, obs, deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        env = self.env
        if env is None:
            raise RuntimeError(
                "GeometricAffineAgent needs an env bound (constructor or bind_env)."
            )
        cfg = self.cfg if self.cfg is not None else env.cfg

        positions, centroid, goal, obstacles = _read_privileged_scene(env)

        # --- heading (same bounded+capped APF as IAPF) → subgoal direction ---
        heading, _att, _rep, _rep_norm, _goal_dist = _apf_heading(
            centroid, goal, obstacles,
            k_att=self.k_att, k_rep=self.k_rep, rho0=self.rho0,
            rep_cap=self.rep_cap, gnron=self.gnron, gnron_r=self.gnron_r,
        )
        dx = float(np.clip(heading[0], -1.0, 1.0))
        dy = float(np.clip(heading[1], -1.0, 1.0))

        # --- theta: align the depth axis with the travel *axis* (wrap to
        #     [-theta_max, theta_max]; a corridor is an undirected line). ---
        ang = float(np.arctan2(heading[1], heading[0]))
        if ang > np.pi / 2:
            ang -= np.pi
        elif ang < -np.pi / 2:
            ang += np.pi
        theta = float(np.clip(ang, -self.theta_max, self.theta_max))
        z_theta = theta / self.theta_max if self.theta_max > 1e-9 else 0.0
        z_theta = float(np.clip(z_theta, -1.0, 1.0))

        # --- s_y: anisotropic cross-path squeeze from the free width ahead ---
        base_half = float(np.max(np.abs(np.asarray(env.formation_offsets)[:, 1])))
        path_dir = heading                              # unit
        path_norm = np.array([-path_dir[1], path_dir[0]])
        look = centroid + self.lookahead * path_dir
        pos_side = np.inf                               # nearest wall on +normal
        neg_side = np.inf                               # nearest wall on -normal
        for opos, orad in obstacles:
            rel = opos - look
            if abs(float(np.dot(rel, path_dir))) > self.along_window:
                continue
            s_perp = float(np.dot(rel, path_norm))
            free = abs(s_perp) - orad                    # surface distance to wall
            if s_perp >= 0.0:
                pos_side = min(pos_side, free)
            else:
                neg_side = min(neg_side, free)
        free_half = min(pos_side, neg_side)             # tightest side bounds fit

        if not np.isfinite(free_half):
            s_y = 1.0                                    # clear ahead → nominal
        else:
            budget = free_half - cfg.vehicle_radius - cfg.d_safe - self.fit_margin
            s_y_fit = budget / base_half if base_half > 1e-9 else 1.0
            # only ever shrink (cap at nominal), floor at s_min
            s_y = float(np.clip(s_y_fit, cfg.s_min, 1.0))

        z_sx = 0.0                                       # s_x = nominal 1.0
        z_sy = 2.0 * s_y - 2.0                           # invert s = 1 + 0.5·z
        z_kappa = 0.0                                    # no shear (v1)

        action = np.array(
            [dx, dy, z_theta, z_sx, float(np.clip(z_sy, -1.0, 1.0)), z_kappa],
            dtype=np.float32,
        )
        return action, None
