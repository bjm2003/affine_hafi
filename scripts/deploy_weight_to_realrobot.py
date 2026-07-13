"""
Copy trained checkpoint from experiments/ to D:/rl_mpc_deploy/models/.

Usage:
    python scripts/deploy_weight_to_realrobot.py \
        --run experiments/run_20260901_143022 \
        --deploy_root D:/rl_mpc_deploy \
        --alias affine_hafi_v1

Result:
    D:/rl_mpc_deploy/models/affine_hafi_v1.zip

Then on the real robot:
    python nodes/single_vehicle_node.py \
        --mode rl_mpc \
        --model_path models/affine_hafi_v1.zip \
        --goal_x 3.0 --goal_y 0.0 --fix_yaw
"""

from __future__ import annotations
import argparse
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True, help="Path to experiment run dir")
    p.add_argument("--deploy_root", type=str, default="D:/rl_mpc_deploy")
    p.add_argument("--alias", type=str, required=True, help="Model alias (e.g., affine_hafi_v1)")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    src = os.path.join(args.run, "best_model.zip")
    dst = os.path.join(args.deploy_root, "models", f"{args.alias}.zip")

    if not os.path.isfile(src):
        raise FileNotFoundError(f"No best_model.zip in {args.run}")

    if not os.path.isdir(os.path.join(args.deploy_root, "models")):
        raise FileNotFoundError(f"Deploy models/ dir not found: {args.deploy_root}/models")

    print(f"[Deploy] Copy:")
    print(f"  {src}")
    print(f"  → {dst}")

    if args.dry_run:
        print("[Deploy] --dry_run, no file copied")
    else:
        shutil.copy2(src, dst)
        print(f"[Deploy] Done. Verify on robot with:")
        print(f"  python nodes/single_vehicle_node.py --mode rl_mpc --model_path models/{args.alias}.zip ...")


if __name__ == "__main__":
    main()
