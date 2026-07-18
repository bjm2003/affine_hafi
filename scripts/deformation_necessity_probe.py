"""Deformation-necessity probe: does OPTIMAL affine beat OPTIMAL isotropic?

The Gate G2 pilot showed privileged Geom == privileged IAPF on all 4 killer
scenarios, i.e. hand-crafted deformation never beat hand-crafted isotropic there.
The only genuine affine win was s_corridor (+30%) IN TRAINING -- but that could be
a learned-policy quirk, not a real mechanism advantage.

This probe isolates the mechanism: run BOTH privileged controllers (no learning,
ground-truth map) on the corridor family. If Geom (affine) >> IAPF (isotropic) on
s_corridor/z_corridor, the deformation mechanism is REAL and we just need killer
scenarios that expose it. If Geom ~= IAPF everywhere, even an oracle affine can't
beat oracle isotropic -> the idea lacks a demonstrable mechanism and needs rethink.

Also logs how much Geom actually deforms (|1 - s_y| squeeze, |theta| rotation) so
a null result can't be blamed on a timid hand-crafted rule.

Usage:
    python scripts/deformation_necessity_probe.py --scenarios s_corridor z_corridor corridor --n_eps 30
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


def run_agent(agent_name, scenario, cfg, n_eps, seed):
    from envs.formation_env import FormationEnv
    action_type = "affine_6d" if agent_name == "geometric_affine" else "hafi_3d"
    enable_proj = agent_name == "geometric_affine"
    env = FormationEnv(cfg=cfg, action_type=action_type, scenario_mode=scenario,
                       enable_projection=enable_proj, seed=seed)
    if agent_name == "geometric_affine":
        from baselines import GeometricAffineAgent
        agent = GeometricAffineAgent(cfg=cfg, env=env)
    else:
        from baselines import IAPFAgent
        agent = IAPFAgent(cfg=cfg, env=env)

    succ = 0
    squeeze_max, theta_max = [], []       # only meaningful for geometric_affine
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed + i)
        agent.bind_env(env)
        done = False
        ep_squeeze, ep_theta = 0.0, 0.0
        while not done:
            a, _ = agent.predict(obs, deterministic=True)
            if agent_name == "geometric_affine":
                s_y = 1.0 + 0.5 * float(a[4])          # z_sy = 2*s_y - 2
                theta = float(a[2]) * cfg.affine_theta_max
                ep_squeeze = max(ep_squeeze, abs(1.0 - s_y))
                ep_theta = max(ep_theta, abs(theta))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        if info.get("success"):
            succ += 1
        squeeze_max.append(ep_squeeze)
        theta_max.append(ep_theta)
    env.close()
    return (succ / n_eps,
            float(np.mean(squeeze_max)),
            float(np.mean(theta_max)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+",
                   default=["s_corridor", "z_corridor", "corridor"])
    p.add_argument("--n_eps", type=int, default=30)
    p.add_argument("--seed", type=int, default=8000)
    args = p.parse_args()

    cfg = Config()
    print(f"[deform-probe] privileged Geom(affine) vs IAPF(isotropic), n_eps={args.n_eps}")
    print(f"[deform-probe] question: does OPTIMAL affine beat OPTIMAL isotropic?\n")
    print(f"{'scenario':<14} {'IAPF SR':>8} {'Geom SR':>8} {'gap':>7}  "
          f"{'Geom squeeze':>13} {'Geom |theta|':>13}")
    print("-" * 72)

    rows = []
    for scen in args.scenarios:
        sr_i, _, _ = run_agent("iapf", scen, cfg, args.n_eps, args.seed)
        sr_g, sq_g, th_g = run_agent("geometric_affine", scen, cfg, args.n_eps, args.seed)
        gap = sr_g - sr_i
        rows.append((scen, sr_i, sr_g, gap, sq_g, th_g))
        print(f"{scen:<14} {sr_i*100:>7.1f}% {sr_g*100:>7.1f}% {gap*100:>+6.1f}%  "
              f"{sq_g:>12.3f} {th_g:>12.3f}")
    print("-" * 72)

    max_gap = max(r[3] for r in rows)
    deforms = any(r[4] > 0.05 or r[5] > 0.1 for r in rows)
    print()
    if max_gap >= 0.15:
        print(f"[deform-probe] VERDICT: optimal affine beats optimal isotropic by "
              f"{max_gap*100:.0f}% somewhere → the deformation MECHANISM is real.\n"
              f"               The Gate G2 killer scenarios were just the wrong ones;\n"
              f"               build killers from the winning geometry.")
    elif not deforms:
        print("[deform-probe] VERDICT: INCONCLUSIVE — Geom barely deformed (squeeze~0, "
              "theta~0),\n               so a null gap may be the hand-crafted rule's fault, "
              "not the idea's.\n               Inspect / strengthen GeometricAffineAgent first.")
    else:
        print(f"[deform-probe] VERDICT: Geom DID deform but still ~= IAPF (max gap "
              f"{max_gap*100:+.0f}%).\n               Even an oracle affine can't beat oracle "
              "isotropic on the corridor\n               family → the idea lacks a demonstrable "
              "mechanism here. Rethink scenarios or thesis.")


if __name__ == "__main__":
    main()
