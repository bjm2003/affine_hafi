"""Learned-affine deformation-usage probe (the decisive A-vs-B test).

The privileged Geom controller LOSES to isotropic IAPF on the corridor family
(deformation_necessity_probe.py), yet the LEARNED affine policy beats HAFI on
s_corridor (66% vs 36%). Two hypotheses:

    A (thesis holds): the learned policy deforms SMARTLY (purposeful anisotropy /
      shear / rotation, triggered by tight scenarios) — smarter than the crude
      hand rule. Then the paper story is "affine deformation must be LEARNED."
    B (thesis decorative): the learned policy stays ~isotropic (s_x ~= s_y,
      theta~0, kappa~0) and wins by better navigation/scale. Then the 4 extra
      affine DOF are unused and the "affine manifold" framing is unsupported.

This probe decodes every action the learned policy emits and measures how much it
actually deforms, split by scenario. The tell: does it use MORE anisotropy in the
tight corridors than in open? And does |s_x - s_y| / |kappa| / |theta| rise above
noise at all?

    anisotropy = |s_x - s_y|   (0 = isotropic; HAFI physically cannot make this >0)
    shear      = |kappa|
    rotation   = |theta| (rad)

Usage (Ubuntu, fast):
    python scripts/affine_usage_probe.py \
        --ckpt experiments/run_20260717_164321_affine/best_model/best_model.zip \
        --scenarios open s_corridor z_corridor corridor --n_eps 30
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


def decode(z, cfg):
    theta = float(np.clip(z[2], -1, 1)) * cfg.affine_theta_max
    s_x = cfg.s_min + (float(np.clip(z[3], -1, 1)) + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)
    s_y = cfg.s_min + (float(np.clip(z[4], -1, 1)) + 1.0) * 0.5 * (cfg.s_max - cfg.s_min)
    kappa = float(np.clip(z[5], -1, 1)) * cfg.affine_kappa_max
    return theta, s_x, s_y, kappa


def run(model, scenario, cfg, n_eps, seed):
    from envs.formation_env import FormationEnv
    env = FormationEnv(cfg=cfg, action_type="affine_6d", scenario_mode=scenario,
                       enable_projection=True, seed=seed)
    aniso, shear, rot, scale = [], [], [], []
    aniso_succ, aniso_fail = [], []
    succ = 0
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed + i)
        done = False
        ep_aniso = []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            th, sx, sy, kp = decode(a, cfg)
            aniso.append(abs(sx - sy)); shear.append(abs(kp))
            rot.append(abs(th)); scale.append(np.sqrt(sx * sy))
            ep_aniso.append(abs(sx - sy))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        s = info.get("success", False)
        succ += int(s)
        (aniso_succ if s else aniso_fail).append(float(np.mean(ep_aniso)))
    env.close()
    return {
        "SR": succ / n_eps,
        "aniso_mean": float(np.mean(aniso)), "aniso_max": float(np.max(aniso)),
        "shear_mean": float(np.mean(shear)), "shear_max": float(np.max(shear)),
        "rot_mean": float(np.mean(rot)), "rot_max": float(np.max(rot)),
        "scale_mean": float(np.mean(scale)),
        "aniso_succ": float(np.mean(aniso_succ)) if aniso_succ else float("nan"),
        "aniso_fail": float(np.mean(aniso_fail)) if aniso_fail else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scenarios", nargs="+",
                   default=["open", "s_corridor", "z_corridor", "corridor"])
    p.add_argument("--n_eps", type=int, default=30)
    p.add_argument("--seed", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = Config()
    from stable_baselines3 import PPO
    model = PPO.load(args.ckpt, device=args.device)

    print(f"[affine-usage] ckpt={args.ckpt}")
    print(f"[affine-usage] s_min={cfg.s_min} s_max={cfg.s_max} "
          f"theta_max={cfg.affine_theta_max:.3f} kappa_max={cfg.affine_kappa_max}")
    print(f"[affine-usage] anisotropy=|s_x-s_y| (max possible={cfg.s_max-cfg.s_min:.2f}); "
          f"isotropic policy => ~0\n")
    print(f"{'scenario':<12} {'SR':>6} {'aniso_mean':>11} {'aniso_max':>10} "
          f"{'shear_mean':>11} {'rot_mean':>9} {'scale':>6}  {'aniso S/F':>12}")
    print("-" * 92)
    rows = []
    for scen in args.scenarios:
        m = run(model, scen, cfg, args.n_eps, args.seed)
        rows.append((scen, m))
        print(f"{scen:<12} {m['SR']*100:>5.0f}% {m['aniso_mean']:>11.3f} "
              f"{m['aniso_max']:>10.3f} {m['shear_mean']:>11.3f} {m['rot_mean']:>9.3f} "
              f"{m['scale_mean']:>6.2f}  {m['aniso_succ']:>5.3f}/{m['aniso_fail']:<5.3f}")
    print("-" * 92)

    # Verdict heuristic: is corridor anisotropy meaningfully above open's, and above noise?
    d = {s: m for s, m in rows}
    open_a = d.get("open", {}).get("aniso_mean", 0.0)
    corr_a = max(d.get(s, {}).get("aniso_mean", 0.0)
                 for s in ("s_corridor", "z_corridor", "corridor") if s in d) if len(d) > 1 else 0.0
    print()
    if corr_a < 0.05:
        print("[affine-usage] VERDICT: B (decorative). Corridor anisotropy ~0 — the policy "
              "barely\n               uses s_x!=s_y. The affine DOF are NOT the mechanism; the "
              "win is\n               navigation/scale. 'Affine manifold' framing unsupported "
              "as-is.")
    elif corr_a > open_a * 1.5 and corr_a > 0.10:
        print("[affine-usage] VERDICT: A (thesis holds). Policy deforms MORE in tight "
              "corridors than\n               in open, above noise. Deformation is purposeful & "
              "scenario-triggered.\n               Story: affine deformation must be LEARNED "
              "(naive hand rule fails).")
    else:
        print("[affine-usage] VERDICT: WEAK/mixed. Some deformation but not clearly "
              "scenario-triggered.\n               Inspect per-episode traces before committing "
              "to a paper claim.")


if __name__ == "__main__":
    main()
