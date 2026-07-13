"""
Pilot: run HAFI baseline and Affine method on 4 killer scenarios.

Purpose (M2 stop-loss Gate G2):
    Verify that Affine method achieves SR gap ≥ 15% over HAFI baseline on:
        1. curved_slot     (弯曲窄廊)
        2. sequential_doorways (门框序列)
        3. asymmetric_density  (侧向不对称)
        4. interior_injection  (中期障碍注入)

If gap < 5% → direction A has fundamental issue, switch to B2 as primary.
If gap 5-15% → analyze cause (emergent behavior not appearing? add entropy reg).
If gap ≥ 15% → proceed to M3 (full training + baseline reproduction).

Usage:
    python scripts/pilot_killer_scenarios.py --method hafi_baseline --scenario curved_slot
    python scripts/pilot_killer_scenarios.py --method affine_hafi   --scenario curved_slot
    python scripts/pilot_killer_scenarios.py --compare  # runs both, prints SR table

TODO:
    - Implement 4 scenario generators in envs/scenario_generators/
    - Wire up FormationEnv with each action_type
    - Load a lightly-trained checkpoint for each method (or run short training here)
    - Aggregate SR / CR / TR over N_trials (default 50)
    - Print result table to stdout, save CSV to experiments/pilot_YYYYMMDD/
"""

from __future__ import annotations
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["hafi_baseline", "affine_hafi"], required=False)
    p.add_argument("--scenario", choices=[
        "curved_slot", "sequential_doorways", "asymmetric_density", "interior_injection"
    ], required=False)
    p.add_argument("--compare", action="store_true", help="Run all methods on all scenarios")
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[TODO] Pilot runner: method={args.method}, scenario={args.scenario}")
    print("This is a stub. Implement after M1 (env + HAFI baseline).")


if __name__ == "__main__":
    main()
