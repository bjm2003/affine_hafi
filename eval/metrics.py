"""
Evaluation metrics aligned with HAFI paper Table 1.

Metrics:
    - SR  Success Rate:                     fraction of episodes reaching goal
    - CR  Collision Rate:                   fraction with any collision
    - TR  Timeout Rate:                     fraction that hit max_episode_steps
    - FE  Formation Error (cm):             mean over successful ep of avg-per-step
                                            pairwise-formation-error (converted to cm)
    - ACT Average Completion Time (s):      mean episode length over successful ep,
                                            converted to seconds via dt_rl
    - EC  Energy Cost:                      mean sum of |u| over successful ep
                                            (proxy for control effort)

Each episode record is a dict with:
    - success:                bool
    - collision:              bool
    - truncated:              bool
    - episode_length:         int
    - formation_error_hist:   list[float]     per-step pairwise formation error (m)
    - control_effort:         float           sum |u| over episode
    - scenario_name:          str
    - mpc_feasibility_rate:   float           avg per-step feasibility
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np


def summarize_episodes(
    episodes: List[Dict[str, Any]],
    dt_rl: float = 0.5,
) -> Dict[str, float]:
    """Aggregate a list of episode records into metric dict."""
    n = len(episodes)
    if n == 0:
        return {
            "n": 0, "SR": 0.0, "CR": 0.0, "TR": 0.0,
            "FE_cm": float("nan"), "ACT_s": float("nan"), "EC": float("nan"),
            "MPC_feasibility": float("nan"),
        }

    n_success = sum(1 for e in episodes if e["success"])
    n_coll = sum(1 for e in episodes if e["collision"])
    n_trunc = sum(1 for e in episodes if e.get("truncated", False))

    successful = [e for e in episodes if e["success"]]
    if successful:
        # FE: mean per-episode-avg formation error (in cm)
        fe_per_ep = []
        for e in successful:
            hist = e.get("formation_error_hist", [])
            if len(hist) > 0:
                fe_per_ep.append(np.mean(hist) * 100.0)
        fe_cm = float(np.mean(fe_per_ep)) if fe_per_ep else float("nan")

        act_s = float(np.mean([e["episode_length"] for e in successful])) * dt_rl
        ec = float(np.mean([e.get("control_effort", float("nan")) for e in successful]))
    else:
        fe_cm = float("nan")
        act_s = float("nan")
        ec = float("nan")

    mpc_feas = float(np.mean([e.get("mpc_feasibility_rate", 1.0) for e in episodes]))

    return {
        "n": n,
        "SR": n_success / n,
        "CR": n_coll / n,
        "TR": n_trunc / n,
        "FE_cm": fe_cm,
        "ACT_s": act_s,
        "EC": ec,
        "MPC_feasibility": mpc_feas,
    }


def summarize_by_scenario(
    episodes: List[Dict[str, Any]],
    dt_rl: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Return {scenario_name: metrics_dict}."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for e in episodes:
        name = e.get("scenario_name", "unknown")
        buckets.setdefault(name, []).append(e)

    return {name: summarize_episodes(eps, dt_rl) for name, eps in buckets.items()}


def format_table(
    tier_results: Dict[str, Dict[str, float]],
    paper_reference: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """Return a human-readable markdown table.

    tier_results: {tier_name: metric_dict}
    paper_reference: optional {tier_name: metric_dict} for side-by-side comparison
    """
    header = "| Tier | n | SR (%) | CR (%) | TR (%) | FE (cm) | ACT (s) | EC | MPC feas (%) |"
    sep = "|------|---|--------|--------|--------|---------|---------|----|--------------|"
    lines = [header, sep]

    for tier, m in tier_results.items():
        row = (
            f"| {tier} | {m['n']} | {m['SR']*100:.1f} | {m['CR']*100:.1f} | "
            f"{m['TR']*100:.1f} | {m['FE_cm']:.2f} | {m['ACT_s']:.2f} | "
            f"{m['EC']:.2f} | {m['MPC_feasibility']*100:.1f} |"
        )
        lines.append(row)

        if paper_reference and tier in paper_reference:
            p = paper_reference[tier]
            lines.append(
                f"| {tier} (paper) | - | {p.get('SR', float('nan'))*100:.1f} | "
                f"{p.get('CR', float('nan'))*100:.1f} | "
                f"{p.get('TR', float('nan'))*100:.1f} | "
                f"{p.get('FE_cm', float('nan')):.2f} | "
                f"{p.get('ACT_s', float('nan')):.2f} | "
                f"{p.get('EC', float('nan')):.2f} | - |"
            )

    return "\n".join(lines)


# HAFI paper Table 1 reference (fractional units)
HAFI_PAPER_REFERENCE = {
    "L1": {"SR": 0.938, "CR": 0.056, "TR": 0.006, "FE_cm": 6.41, "ACT_s": 45.92, "EC": 3.64},
    "L2": {"SR": 0.918, "CR": 0.080, "TR": 0.002, "FE_cm": 6.90, "ACT_s": 47.19, "EC": 3.73},
    "L3": {"SR": 0.830, "CR": 0.164, "TR": 0.006, "FE_cm": 7.50, "ACT_s": 49.64, "EC": 3.85},
}
