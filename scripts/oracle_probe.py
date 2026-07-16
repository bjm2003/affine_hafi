"""
Checkpoint-free solvability probe (Gate G1 root-cause).

Runs a *perfect* privileged oracle policy — greedy heading straight at the goal,
nominal scale — through the env's real MPC substep loop. This isolates two very
different failure classes for the timeout-dominated Gate G1 miss:

    * oracle SUCCEEDS on open  → env is solvable; the trained policy's failure is
      a LEARNING / REWARD-SHAPING problem (meander, weak goal signal, etc.)
    * oracle FAILS on open     → a MECHANICS / SPEC bug (carrot too weak, MPC
      stalls near a close subgoal, goal_tolerance too tight, v_max too low) that
      NO amount of retraining can fix.

The oracle reads privileged env state (positions/goal), exactly like the classical
baselines, so it needs no checkpoint. For hafi_3d the greedy action is:
    dx, dy = unit(goal - centroid)      # push the R0 carrot straight at goal
    scale  = 0.0                        # → nominal scale 1.0

Usage:
    python scripts/oracle_probe.py --scenarios open corridor --n_eps 20
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from envs.formation_env import FormationEnv


def oracle_action(env, action_type):
    center = env.positions.mean(axis=0)
    g = env.goal - center
    n = float(np.linalg.norm(g))
    u = g / n if n > 1e-9 else np.array([1.0, 0.0])
    if action_type == "hafi_3d":
        return np.array([u[0], u[1], 0.0], dtype=np.float32)  # scale=0 → 1.0
    # affine_6d: heading + nominal formation (theta/sx/sy/kappa = 0)
    return np.array([u[0], u[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def rollout(env, action_type, seed, gap):
    kwargs = {"gap_width": gap} if gap is not None else None
    obs, _ = env.reset(seed=seed, options=(
        {"scenario_kwargs": kwargs} if kwargs else None))
    centers = [env.positions.mean(axis=0).copy()]
    dists = []
    d0 = float(np.linalg.norm(centers[0] - env.goal))
    outcome, steps = "timeout", 0
    speeds = []
    while True:
        a = oracle_action(env, action_type)
        prev_c = env.positions.mean(axis=0).copy()
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        dists.append(info["dist_to_goal"])
        centers.append(info["center"].copy())
        speeds.append(float(np.linalg.norm(info["center"] - prev_c)) / env.cfg.dt_rl)
        if term:
            outcome = "success" if info["success"] else "collision"
            break
        if trunc:
            outcome = "timeout"
            break
    centers = np.asarray(centers)
    path_len = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    net = float(np.linalg.norm(centers[0] - centers[-1]))
    return {
        "outcome": outcome, "steps": steps, "d0": d0,
        "d_final": dists[-1], "d_min": float(np.min(dists)),
        "path_eff": net / path_len if path_len > 1e-9 else 0.0,
        "mean_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "last8_dist": [round(x, 3) for x in dists[-8:]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=["open", "corridor"])
    p.add_argument("--action_type", default="hafi_3d",
                   choices=["hafi_3d", "affine_6d"])
    p.add_argument("--n_eps", type=int, default=20)
    p.add_argument("--gap", type=float, default=None)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    cfg = Config()
    print(f"ORACLE probe  action_type={args.action_type}  "
          f"v_max={cfg.v_max} goal_tol={cfg.goal_tolerance} "
          f"max_steps={cfg.max_episode_steps} dt_rl={cfg.dt_rl}")
    print(f"  straight-line budget: max dist @ v_max in {cfg.max_episode_steps} steps "
          f"= {cfg.v_max * cfg.dt_rl * cfg.max_episode_steps:.1f} m "
          f"(start range {cfg.start_distance_range})\n")

    for scen in args.scenarios:
        enable_proj = (args.action_type == "affine_6d")
        env = FormationEnv(cfg=cfg, action_type=args.action_type,
                           scenario_mode=scen, enable_projection=enable_proj,
                           seed=args.seed)
        rs = [rollout(env, args.action_type, seed=args.seed + i,
                      gap=(args.gap if scen == "corridor" else None))
              for i in range(args.n_eps)]
        env.close()
        succ = [r for r in rs if r["outcome"] == "success"]
        tmo = [r for r in rs if r["outcome"] == "timeout"]
        coll = [r for r in rs if r["outcome"] == "collision"]
        sr = len(succ) / len(rs)
        print(f"=== {scen} (gap={args.gap})  ORACLE SR={sr*100:.0f}%  "
              f"timeout={len(tmo)} collision={len(coll)} ===")
        print(f"  steps    succ={np.mean([r['steps'] for r in succ]) if succ else float('nan'):.0f}"
              f"  (budget {cfg.max_episode_steps})")
        print(f"  speed    mean={np.mean([r['mean_speed'] for r in rs]):.3f}"
              f"  max={np.mean([r['max_speed'] for r in rs]):.3f}"
              f"  (v_max={cfg.v_max})  → speed/v_max={np.mean([r['mean_speed'] for r in rs])/cfg.v_max*100:.0f}%")
        print(f"  path_eff mean={np.mean([r['path_eff'] for r in rs]):.2f} (1.0=straight)")
        ex = (tmo or rs)[0]
        print(f"  [{ex['outcome']} ex] d0={ex['d0']:.2f} d_min={ex['d_min']:.2f} "
              f"d_final={ex['d_final']:.2f} steps={ex['steps']} "
              f"mean_speed={ex['mean_speed']:.3f}")
        print(f"     last8 dist: {ex['last8_dist']}")
        print()


if __name__ == "__main__":
    main()
