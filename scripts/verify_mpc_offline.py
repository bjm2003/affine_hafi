"""
Verify MPC solver copied from D:/rl_mpc_deploy/ still works on this machine.

This is a sanity check for M1: solver JIT compiles, IPOPT solves, feasible
result on a trivial problem.

Usage:
    python scripts/verify_mpc_offline.py
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from mpc.solver import MPCSolver


def main():
    print("[Verify] Constructing MPCSolver ...")
    cfg = Config()
    t0 = time.perf_counter()
    solver = MPCSolver(cfg)
    print(f"[Verify] MPC ready in {time.perf_counter() - t0:.2f}s")

    # Trivial problem: current at origin, goal at (1, 0), no obstacles
    current_pos = np.array([0.0, 0.0])
    ref_pos = np.array([1.0, 0.0])

    print("[Verify] Solving trivial problem (goal at (1, 0), no obstacles) ...")
    t0 = time.perf_counter()
    u, feasible, slack, status = solver.solve(
        current_pos=current_pos,
        ref_pos=ref_pos,
        obstacles=[],
        other_positions=[],
        current_scale=1.0,
    )
    dt = time.perf_counter() - t0
    print(f"  u = {u}")
    print(f"  feasible = {feasible}")
    print(f"  slack_sum = {slack:.6f}")
    print(f"  status = {status}")
    print(f"  solve time = {dt * 1000:.2f} ms")

    if feasible and np.linalg.norm(u) > 0.01:
        print("\n[Verify] PASS ✓")
    else:
        print("\n[Verify] FAIL — MPC returned infeasible or zero velocity")
        sys.exit(1)


if __name__ == "__main__":
    main()
