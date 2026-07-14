"""
Main evaluation script. Loads a trained model and evaluates on L1/L2/L3 tiers.

Tier definitions (matches HAFI paper Table 1 L1/L2/L3):
    L1 easy:    scenarios that stress simple avoidance & progress
                → open, corridor (gap≥0.85), dynamic (n_dyn=1)
    L2 medium:  more complex geometry
                → s_corridor, z_corridor, dynamic (default)
    L3 hard:    combinations that stress everything
                → u_trap, corridor (gap=0.70), dynamic (n_dyn=2)

For each tier: N episodes per scenario, deterministic policy, fixed seeds.

Usage:
    python eval/eval.py --run experiments/run_YYYYMMDD_HHMMSS
    python eval/eval.py --run experiments/... --n_per_scenario 50 --tiers L1,L2
    python eval/eval.py --checkpoint experiments/.../best_model/best_model.zip
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============ Tier definitions ============
TIER_SPEC = {
    "L1": [
        {"scenario": "open", "kwargs": {}, "weight": 1.0},
        {"scenario": "corridor", "kwargs": {"gap_width": 0.90}, "weight": 1.0},
        {"scenario": "dynamic", "kwargs": {}, "weight": 1.0},
    ],
    "L2": [
        {"scenario": "s_corridor", "kwargs": {"gap_width": 0.85}, "weight": 1.0},
        {"scenario": "z_corridor", "kwargs": {"gap_width": 0.85}, "weight": 1.0},
        {"scenario": "dynamic", "kwargs": {}, "weight": 1.0},
    ],
    "L3": [
        {"scenario": "u_trap", "kwargs": {}, "weight": 1.0},
        {"scenario": "corridor", "kwargs": {"gap_width": 0.70}, "weight": 1.0},
        {"scenario": "dynamic", "kwargs": {}, "weight": 1.0},
    ],
    # Killer scenarios (for M2 pilot / final Table)
    "KILLER": [
        {"scenario": "curved_slot", "kwargs": {}, "weight": 1.0},
        {"scenario": "sequential_doorways", "kwargs": {}, "weight": 1.0},
        {"scenario": "asymmetric_density", "kwargs": {}, "weight": 1.0},
        {"scenario": "interior_injection", "kwargs": {}, "weight": 1.0},
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None,
                   help="Experiment dir (uses best_model/best_model.zip by default)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit path to .zip checkpoint (overrides --run)")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config path; defaults to <run>/config_snapshot.yaml")
    p.add_argument("--tiers", type=str, default="L1,L2,L3",
                   help="Comma-separated tier names to eval (subset of {L1,L2,L3,KILLER})")
    p.add_argument("--n_per_scenario", type=int, default=50,
                   help="Episodes per scenario within each tier")
    p.add_argument("--seed_base", type=int, default=10_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path; defaults to <run>/eval_<tiers>.json")
    return p.parse_args()


def resolve_checkpoint(args) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.run:
        run = Path(args.run)
        best = run / "best_model" / "best_model.zip"
        if best.is_file():
            return best
        final = run / "final_model.zip"
        if final.is_file():
            return final
        # Fall back to latest checkpoint
        ckpts = sorted((run / "checkpoints").glob("rl_model_*_steps.zip"))
        if ckpts:
            return ckpts[-1]
        raise FileNotFoundError(f"No checkpoint found in {run}")
    raise ValueError("Must specify --run or --checkpoint")


def resolve_config(args) -> str:
    if args.config:
        return args.config
    if args.run:
        snap = Path(args.run) / "config_snapshot.yaml"
        if snap.is_file():
            return str(snap)
    raise FileNotFoundError("No --config specified and no snapshot found in --run dir")


def main():
    args = parse_args()

    from stable_baselines3 import PPO
    from envs.formation_env import FormationEnv
    from train.config_loader import load_config
    from eval.eval_runner import run_batch
    from eval.metrics import summarize_episodes, summarize_by_scenario, format_table, HAFI_PAPER_REFERENCE

    # ============ Load model + config ============
    ckpt_path = resolve_checkpoint(args)
    cfg_path = resolve_config(args)
    print(f"[Eval] Checkpoint: {ckpt_path}")
    print(f"[Eval] Config: {cfg_path}")

    cfg, extras = load_config(cfg_path)
    method_cfg = extras.get("method", {})
    action_type = str(method_cfg.get("action_type", "hafi_3d"))

    # Need to register custom policy classes for PPO.load
    from policies.feature_extractor import FormationFeatureExtractor  # noqa
    if action_type == "hafi_3d":
        from policies.dual_head_policy import DualHeadPolicy  # noqa
    elif action_type == "affine_6d":
        from policies.affine_policy import AffinePolicy  # noqa

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = PPO.load(str(ckpt_path), custom_objects=custom_objects, device=args.device)
    print(f"[Eval] Model loaded ({action_type})")

    # ============ Build eval env ============
    env = FormationEnv(
        cfg=cfg,
        action_type=action_type,
        scenario_mode="mixed",   # will be overridden per-episode via options
        entropy_weight=float(method_cfg.get("formation_entropy_weight", 0.0)),
        affine_theta_max=float(method_cfg.get("affine_theta_max", 1.5708)),
        affine_kappa_max=float(method_cfg.get("affine_kappa_max", 0.5)),
    )
    # Disable world clip during eval so we accept scenarios that place goals
    # slightly outside the training arena
    env.cfg.enable_world_clip = False

    # ============ Run tiers ============
    tier_names = [t.strip() for t in args.tiers.split(",") if t.strip()]
    all_results: Dict[str, Any] = {}
    tier_summary: Dict[str, Dict[str, float]] = {}

    for tier in tier_names:
        if tier not in TIER_SPEC:
            print(f"[Eval] WARN: unknown tier '{tier}', skipping")
            continue
        print(f"\n[Eval] === Tier {tier} ===")
        tier_records = []
        for spec in TIER_SPEC[tier]:
            scenario = spec["scenario"]
            scen_kwargs = spec["kwargs"]
            print(f"  [{tier}/{scenario}] running {args.n_per_scenario} eps...")
            recs = run_batch(
                env, model,
                n_episodes=args.n_per_scenario,
                deterministic=True,
                seed_base=args.seed_base,
                scenario_override=scenario,
                scenario_kwargs=scen_kwargs if scen_kwargs else None,
            )
            tier_records.extend(recs)
            sub = summarize_episodes(recs, dt_rl=cfg.dt_rl)
            print(f"    SR={sub['SR']*100:.1f}%  CR={sub['CR']*100:.1f}%  "
                  f"TR={sub['TR']*100:.1f}%  ACT={sub['ACT_s']:.2f}s "
                  f"MPCfeas={sub['MPC_feasibility']*100:.1f}%")

        tier_summary[tier] = summarize_episodes(tier_records, dt_rl=cfg.dt_rl)
        by_scenario = summarize_by_scenario(tier_records, dt_rl=cfg.dt_rl)
        all_results[tier] = {
            "aggregate": tier_summary[tier],
            "by_scenario": by_scenario,
            "n_episodes": len(tier_records),
        }

    # ============ Print comparison table ============
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(format_table(tier_summary, paper_reference=HAFI_PAPER_REFERENCE))
    print("=" * 70)

    # ============ Gate G1 check ============
    if "L1" in tier_summary:
        l1_sr = tier_summary["L1"]["SR"]
        target = 0.888
        target_strong = 0.938
        if l1_sr >= target_strong:
            print(f"\n[Gate G1] ★ PASS (strong): L1 SR = {l1_sr*100:.1f}% ≥ {target_strong*100:.1f}%")
        elif l1_sr >= target:
            print(f"\n[Gate G1] ✓ PASS: L1 SR = {l1_sr*100:.1f}% ≥ {target*100:.1f}%")
        else:
            print(f"\n[Gate G1] ✗ FAIL: L1 SR = {l1_sr*100:.1f}% < {target*100:.1f}%")
            print("[Gate G1] See docs/decisions.md → Phase D debug playbook")

    # ============ Save JSON ============
    if args.output:
        out_path = Path(args.output)
    elif args.run:
        tag = "_".join(tier_names)
        out_path = Path(args.run) / f"eval_{tag}.json"
    else:
        out_path = Path("eval_results.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(ckpt_path),
            "config": str(cfg_path),
            "action_type": action_type,
            "tiers": all_results,
            "cli": {
                "tiers": tier_names,
                "n_per_scenario": args.n_per_scenario,
                "seed_base": args.seed_base,
            },
        }, f, indent=2, default=float)
    print(f"\n[Eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()
