"""U-trap affine solvability ceiling probe.

Question: can the PRIVILEGED geometric-affine controller (ground-truth obstacle
map, optimal shear/rotate/squeeze rule) escape a u_trap? This establishes the
affine solvability ceiling for u_trap the same way oracle_probe did for open.

Rationale: u_trap places the team INSIDE a three-sided box, opening facing away
from the goal. Escaping requires driving BACKWARD out the opening then detouring
around -- a heading/planning decision that lives in (dx, dy), which HAFI and the
affine method share identically. The affine extra DOF (theta, s_x, s_y, kappa)
only reshape the formation. If the privileged affine controller scores ~0, then
u_trap does NOT exercise the affine advantage, and RL failing to learn it during
training is EXPECTED, not a bug to fix.

Compare against IAPF (privileged isotropic APF, hafi_3d) on the same u_trap: if
both score ~0, the two methods are mechanically equivalent on this scenario.

Usage:
    python scripts/u_trap_affine_probe.py --n_eps 30
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


def run_agent(agent_name, cfg, n_eps, seed):
    action_type = "affine_6d" if agent_name == "geometric_affine" else "hafi_3d"
    enable_proj = agent_name == "geometric_affine"
    env = FormationEnv(cfg=cfg, action_type=action_type, scenario_mode="u_trap",
                       enable_projection=enable_proj, seed=seed)
    if agent_name == "geometric_affine":
        from baselines import GeometricAffineAgent
        agent = GeometricAffineAgent(cfg=cfg, env=env)
    else:
        from baselines import IAPFAgent
        agent = IAPFAgent(cfg=cfg, env=env)

    succ, min_dists = 0, []
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed + i)
        agent.bind_env(env)
        done = False
        d_min = np.inf
        while not done:
            a, _ = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            center = np.asarray(env.positions, dtype=np.float64).mean(axis=0)
            d_min = min(d_min, float(np.linalg.norm(center - env.goal)))
            done = term or trunc
        if info.get("success"):
            succ += 1
        min_dists.append(d_min)
    env.close()
    return succ / n_eps, float(np.mean(min_dists))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eps", type=int, default=30)
    p.add_argument("--seed", type=int, default=7000)
    args = p.parse_args()

    cfg = Config()
    print(f"[u_trap-probe] privileged controllers on u_trap, n_eps={args.n_eps}\n")

    sr_g, dmin_g = run_agent("geometric_affine", cfg, args.n_eps, args.seed)
    print(f"[u_trap-probe] GeometricAffine (affine_6d, proj ON)  SR={sr_g*100:5.1f}%  "
          f"mean_closest_approach={dmin_g:.2f}m")

    sr_i, dmin_i = run_agent("iapf", cfg, args.n_eps, args.seed)
    print(f"[u_trap-probe] IAPF          (hafi_3d,  proj OFF) SR={sr_i*100:5.1f}%  "
          f"mean_closest_approach={dmin_i:.2f}m")

    print()
    if sr_g < 0.15 and sr_i < 0.15:
        print("[u_trap-probe] VERDICT: BOTH privileged controllers fail u_trap → "
              "u_trap does NOT exercise the affine advantage (it's a detour/planning\n"
              "               scenario, shared heading DOF). RL failing to learn it is\n"
              "               EXPECTED. Remove u_trap from affine TRAINING to stop\n"
              "               failure_replay poisoning; it is NOT an affine killer scenario.")
    elif sr_g - sr_i >= 0.15:
        print(f"[u_trap-probe] VERDICT: affine ceiling ({sr_g*100:.0f}%) >> isotropic "
              f"({sr_i*100:.0f}%) → u_trap DOES reward affine deformation; keep training it.")
    else:
        print("[u_trap-probe] VERDICT: mixed — inspect trajectories before deciding.")


if __name__ == "__main__":
    main()
