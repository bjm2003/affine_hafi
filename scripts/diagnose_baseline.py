"""
Diagnose a trained hafi_3d checkpoint's FAILURE MODE (Gate G1 debug).

Symptom under investigation: timeout-dominated failure (no collision, MPC
feasible) with high control effort + loose formation. This script rolls out the
deterministic policy and classifies WHY episodes end, so we can tell apart:

    * meandering   — centroid path length >> straight-line dist (low path eff.)
    * stall        — dist_to_goal plateaus (stuck, not moving)
    * orbit        — dist_to_goal oscillates near a floor > goal_tolerance
    * corridor-fit — scale stays large in corridors (can't shrink to pass)
    * genuine-slow — efficient but just runs out of step budget

Usage:
    python scripts/diagnose_baseline.py --ckpt <path-to-best_model.zip> \
        --scenarios open corridor dynamic --n_eps 20
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


def rollout(env, model, seed, gap):
    kwargs = {"gap_width": gap} if gap is not None else None
    obs, _ = env.reset(seed=seed, options=(
        {"scenario_kwargs": kwargs} if kwargs else None))
    centers = [env.positions.mean(axis=0).copy()]
    dists, scales, act_mags, comp_sums = [], [], [], {}
    d0 = float(np.linalg.norm(centers[0] - env.goal))
    outcome = "timeout"
    steps = 0
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        dists.append(info["dist_to_goal"])
        scales.append(info["current_scale"])
        act_mags.append(float(np.linalg.norm(np.asarray(a[:2]))))  # heading mag
        centers.append(info["center"].copy())
        for k, v in info["reward_components"].items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
        if term:
            outcome = "success" if info["success"] else "collision"
            break
        if trunc:
            outcome = "timeout"
            break
    centers = np.asarray(centers)
    path_len = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    net = float(np.linalg.norm(centers[0] - centers[-1]))
    path_eff = net / path_len if path_len > 1e-9 else 0.0
    return {
        "outcome": outcome, "steps": steps, "d0": d0,
        "d_final": dists[-1], "d_min": float(np.min(dists)),
        "path_len": path_len, "path_eff": path_eff,
        "scale_mean": float(np.mean(scales)), "scale_min": float(np.min(scales)),
        "heading_mag_mean": float(np.mean(act_mags)),
        "last10_dist": [round(x, 3) for x in dists[-10:]],
        "comp_sums": {k: round(v, 2) for k, v in comp_sums.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scenarios", nargs="+", default=["open", "corridor", "dynamic"])
    p.add_argument("--n_eps", type=int, default=20)
    p.add_argument("--gap", type=float, default=None,
                   help="corridor gap override (e.g. 0.85 for L1, 0.70 for L3)")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    from stable_baselines3 import PPO
    cfg = Config()
    model = PPO.load(args.ckpt, device="cpu")
    print(f"Loaded {args.ckpt}\n")

    for scen in args.scenarios:
        env = FormationEnv(cfg=cfg, action_type="hafi_3d", scenario_mode=scen,
                           enable_projection=False, seed=args.seed)
        rs = [rollout(env, model, seed=args.seed + i,
                      gap=(args.gap if scen == "corridor" else None))
              for i in range(args.n_eps)]
        env.close()
        succ = [r for r in rs if r["outcome"] == "success"]
        tmo = [r for r in rs if r["outcome"] == "timeout"]
        coll = [r for r in rs if r["outcome"] == "collision"]
        sr = len(succ) / len(rs)
        print(f"=== {scen}  (gap={args.gap})  SR={sr*100:.0f}%  "
              f"timeout={len(tmo)} collision={len(coll)} ===")
        print(f"  path_eff  succ={np.mean([r['path_eff'] for r in succ]) if succ else float('nan'):.2f}"
              f"  timeout={np.mean([r['path_eff'] for r in tmo]) if tmo else float('nan'):.2f}"
              f"   (1.0=straight line, low=meander)")
        print(f"  scale     mean={np.mean([r['scale_mean'] for r in rs]):.2f}"
              f"  min={np.mean([r['scale_min'] for r in rs]):.2f}"
              f"   heading_mag={np.mean([r['heading_mag_mean'] for r in rs]):.2f}")
        if tmo:
            t = tmo[0]
            print(f"  [timeout ex] d0={t['d0']:.2f} d_min={t['d_min']:.2f} "
                  f"d_final={t['d_final']:.2f} eff={t['path_eff']:.2f} "
                  f"scale_min={t['scale_min']:.2f}")
            print(f"     last10 dist: {t['last10_dist']}")
            print(f"     reward comp sums: {t['comp_sums']}")
        if succ:
            s = succ[0]
            print(f"  [success ex] steps={s['steps']} eff={s['path_eff']:.2f} "
                  f"comp: {s['comp_sums']}")
        print()


if __name__ == "__main__":
    main()
