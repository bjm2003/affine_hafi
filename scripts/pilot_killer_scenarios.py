"""
Pilot: HAFI baseline vs Affine method on the 4 killer scenarios (M2 Gate G2).

Purpose:
    Verify the Affine method (6D affine intention + feasibility projection)
    achieves a Success-Rate gap ≥ 15% over the HAFI 3D baseline on the four
    scenarios designed to expose the limits of isotropic scaling:
        1. curved_slot          (curved narrow channel)
        2. sequential_doorways  (a chain of offset doorways)
        3. asymmetric_density   (obstacles crowd one side)
        4. interior_injection   (obstacle injected mid-episode)

Gate G2 verdict (on the mean SR gap over the 4 scenarios):
    gap ≥ 0.15   → PASS   → proceed to M3 (full training + baselines)
    0.05–0.15    → MARGIN → analyze (emergent behavior missing? enable entropy reg)
    gap < 0.05   → FAIL   → switch to direction B2 as primary

Usage:
    # Real comparison (needs trained checkpoints from Ubuntu training machine):
    python scripts/pilot_killer_scenarios.py --compare \
        --hafi_ckpt experiments/run_..._hafi/best_model/best_model.zip \
        --affine_ckpt experiments/run_..._affine/best_model/best_model.zip \
        --n_trials 50

    # Harness smoke test with untrained (random) policies:
    python scripts/pilot_killer_scenarios.py --compare --n_trials 5

    # Single cell:
    python scripts/pilot_killer_scenarios.py --method affine_hafi \
        --scenario curved_slot --affine_ckpt <path> --n_trials 20

    # C2 ablation (projection ON vs OFF for the affine method):
    python scripts/pilot_killer_scenarios.py --compare --proj_ablation \
        --affine_ckpt <path> --n_trials 50
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from envs.formation_env import FormationEnv
from eval.eval_runner import run_batch
from eval.metrics import summarize_episodes

KILLER_SCENARIOS = [
    "curved_slot", "sequential_doorways", "asymmetric_density", "interior_injection",
]


class RandomAgent:
    """Fallback agent used when no checkpoint is provided (harness smoke test)."""

    def __init__(self, action_space, seed: int = 0):
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs, deterministic: bool = True):
        return self.action_space.sample(), None


def _action_type(method: str) -> str:
    # Classical IAPF is a drop-in high-level over the same MPC as HAFI → hafi_3d.
    if method in ("hafi_baseline", "iapf"):
        return "hafi_3d"
    # affine_hafi (learned) and geometric_affine (classical) both emit affine_6d.
    return "affine_6d"


def build_env(
    method: str,
    scenario: str,
    enable_projection: bool,
    cfg: Config,
    seed: int,
) -> FormationEnv:
    return FormationEnv(
        cfg=cfg,
        action_type=_action_type(method),
        scenario_mode=scenario,          # every reset picks this killer scenario
        enable_projection=enable_projection,
        seed=seed,
    )


def load_agent(method: str, ckpt: Optional[str], env: FormationEnv, seed: int, cfg: Config):
    """Return the agent for a cell: classical IAPF/geometric, SB3 model, or Random."""
    if method == "iapf":
        from baselines import IAPFAgent
        print(f"    classical IAPF controller (privileged obstacle map, no checkpoint)")
        return IAPFAgent(cfg=cfg, env=env)
    if method == "geometric_affine":
        from baselines import GeometricAffineAgent
        print(f"    geometric affine controller (privileged obstacle map, no checkpoint)")
        return GeometricAffineAgent(cfg=cfg, env=env)
    if ckpt is None:
        print(f"    [warn] no checkpoint → using RandomAgent (harness smoke only)")
        return RandomAgent(env.action_space, seed=seed)
    from stable_baselines3 import PPO
    print(f"    loading checkpoint: {ckpt}")
    return PPO.load(ckpt, device="cpu")


def run_cell(
    method: str,
    scenario: str,
    ckpt: Optional[str],
    enable_projection: bool,
    n_trials: int,
    seed_base: int,
    cfg: Config,
) -> Dict[str, Any]:
    """Run one (method, scenario) cell and return summarized metrics."""
    env = build_env(method, scenario, enable_projection, cfg, seed=seed_base)
    agent = load_agent(method, ckpt, env, seed=seed_base, cfg=cfg)
    records = run_batch(
        env, agent,
        n_episodes=n_trials,
        deterministic=True,
        seed_base=seed_base,
        scenario_override=scenario,
    )
    env.close()
    m = summarize_episodes(records, dt_rl=cfg.dt_rl)
    m["method"] = method
    m["scenario"] = scenario
    m["projection"] = enable_projection
    return m


def verdict(mean_gap: float) -> str:
    if mean_gap >= 0.15:
        return "PASS  → proceed to M3"
    if mean_gap >= 0.05:
        return "MARGIN → analyze (enable entropy reg / inspect emergent behavior)"
    return "FAIL  → switch to direction B2 as primary"


def run_compare(args, cfg: Config) -> Dict[str, Any]:
    scenarios = args.scenarios or KILLER_SCENARIOS
    rows: List[Dict[str, Any]] = []
    gaps: List[float] = []

    print("\n=== M2 Gate G2 pilot: HAFI baseline vs Affine ===")
    for si, scen in enumerate(scenarios):
        seed = args.seed + si * 10_000
        print(f"\n[{scen}]")
        print("  HAFI baseline:")
        hafi = run_cell("hafi_baseline", scen, args.hafi_ckpt,
                        enable_projection=False, n_trials=args.n_trials,
                        seed_base=seed, cfg=cfg)
        print("  Affine (proj ON):")
        affine = run_cell("affine_hafi", scen, args.affine_ckpt,
                          enable_projection=True, n_trials=args.n_trials,
                          seed_base=seed, cfg=cfg)

        gap = affine["SR"] - hafi["SR"]
        gaps.append(gap)
        row = {
            "scenario": scen,
            "hafi_SR": hafi["SR"], "affine_SR": affine["SR"], "gap": gap,
            "hafi_CR": hafi["CR"], "affine_CR": affine["CR"],
            "affine_feas": affine["MPC_feasibility"],
        }

        if args.with_iapf:
            print("  IAPF (classical, proj OFF):")
            iapf = run_cell("iapf", scen, None,
                            enable_projection=False, n_trials=args.n_trials,
                            seed_base=seed, cfg=cfg)
            row["iapf_SR"] = iapf["SR"]
            row["iapf_CR"] = iapf["CR"]

        if args.with_geometric:
            # Classical affine: same affine_6d pipeline as the learned method,
            # projection ON, so the comparison isolates learned vs hand intention.
            print("  Geometric affine (classical, proj ON):")
            geom = run_cell("geometric_affine", scen, None,
                            enable_projection=True, n_trials=args.n_trials,
                            seed_base=seed, cfg=cfg)
            row["geom_SR"] = geom["SR"]
            row["geom_CR"] = geom["CR"]

        if args.proj_ablation:
            print("  Affine (proj OFF, ablation):")
            affine_np = run_cell("affine_hafi", scen, args.affine_ckpt,
                                 enable_projection=False, n_trials=args.n_trials,
                                 seed_base=seed, cfg=cfg)
            row["affine_noproj_SR"] = affine_np["SR"]
            row["affine_noproj_feas"] = affine_np["MPC_feasibility"]
            row["proj_SR_gain"] = affine["SR"] - affine_np["SR"]
        rows.append(row)

    mean_gap = float(np.mean(gaps)) if gaps else 0.0

    # ---- print table ----
    print("\n" + "=" * 90)
    iapf_hdr = f"{'IAPF SR':>8} " if args.with_iapf else ""
    geom_hdr = f"{'Geom SR':>8} " if args.with_geometric else ""
    print(f"{'scenario':<22} {iapf_hdr}{geom_hdr}{'HAFI SR':>8} {'Aff SR':>8} {'gap':>7} "
          f"{'Aff feas':>9}")
    print("-" * 90)
    for r in rows:
        iapf_cell = f"{r['iapf_SR']*100:>7.1f}% " if args.with_iapf else ""
        geom_cell = f"{r['geom_SR']*100:>7.1f}% " if args.with_geometric else ""
        print(f"{r['scenario']:<22} {iapf_cell}{geom_cell}{r['hafi_SR']*100:>7.1f}% "
              f"{r['affine_SR']*100:>7.1f}% {r['gap']*100:>+6.1f}% "
              f"{r['affine_feas']*100:>8.1f}%")
    print("-" * 90)
    print(f"{'MEAN GAP':<22} {'':>8} {'':>8} {mean_gap*100:>+6.1f}%")
    if args.proj_ablation:
        print("\nC2 ablation (projection SR gain, affine ON − OFF):")
        for r in rows:
            print(f"  {r['scenario']:<22} {r.get('proj_SR_gain', 0.0)*100:>+6.1f}%")
    print("=" * 90)
    print(f"Gate G2 verdict: {verdict(mean_gap)}")
    print("=" * 78)

    return {"rows": rows, "mean_gap": mean_gap, "verdict": verdict(mean_gap)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["hafi_baseline", "affine_hafi", "iapf",
                                         "geometric_affine"], default=None)
    p.add_argument("--scenario", choices=KILLER_SCENARIOS, default=None)
    p.add_argument("--scenarios", nargs="*", choices=KILLER_SCENARIOS, default=None,
                   help="Subset of killer scenarios for --compare (default: all 4)")
    p.add_argument("--compare", action="store_true",
                   help="Run HAFI vs Affine over all killer scenarios + Gate G2 verdict")
    p.add_argument("--with_iapf", action="store_true",
                   help="Also run the classical IAPF baseline as a reference column "
                        "(does not affect the Gate G2 gap/verdict)")
    p.add_argument("--with_geometric", action="store_true",
                   help="Also run the classical geometric-affine baseline as a reference "
                        "column (does not affect the Gate G2 gap/verdict)")
    p.add_argument("--proj_ablation", action="store_true",
                   help="Also run affine with projection OFF (C2 ablation)")
    p.add_argument("--hafi_ckpt", type=str, default=None)
    p.add_argument("--affine_ckpt", type=str, default=None)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    stamp = datetime.now().strftime("pilot_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "experiments" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        result = run_compare(args, cfg)
    else:
        if not (args.method and args.scenario):
            print("Provide --compare, or both --method and --scenario.")
            return
        ckpt = args.hafi_ckpt if args.method == "hafi_baseline" else args.affine_ckpt
        if args.method in ("iapf", "geometric_affine"):
            ckpt = None
        # Affine pipeline (learned or geometric) runs with the C2 projection ON.
        enable_proj = args.method in ("affine_hafi", "geometric_affine")
        m = run_cell(args.method, args.scenario, ckpt, enable_proj,
                     args.n_trials, args.seed, cfg)
        print(json.dumps(m, indent=2))
        result = m

    out_path = out_dir / "pilot_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
