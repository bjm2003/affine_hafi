#!/usr/bin/env bash
# HAFI baseline training launcher (Ubuntu, in-repo).
#
# Usage:
#   bash scripts/launch_hafi_baseline.sh                 # 12 envs, GPU, 1.5M steps
#   N_ENVS=8 bash scripts/launch_hafi_baseline.sh        # override n_envs
#   TOTAL=200000 bash scripts/launch_hafi_baseline.sh    # smoke run
#   DEVICE=cpu bash scripts/launch_hafi_baseline.sh      # CPU fallback
#
# Env vars respected:
#   N_ENVS      (default 12)
#   TOTAL       (default from config, typically 1_500_000)
#   DEVICE      (default cuda)
#   SEED        (default 42)
#   EXP_NAME    (default auto timestamp)
#   CONFIG      (default configs/baselines/hafi_original.yaml)

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate affine_hafi

N_ENVS=${N_ENVS:-12}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}
CONFIG=${CONFIG:-configs/baselines/hafi_original.yaml}
STAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME=${EXP_NAME:-run_${STAMP}_hafi_baseline}

TOTAL_ARG=""
if [[ -n "${TOTAL:-}" ]]; then
    TOTAL_ARG="--total_timesteps $TOTAL"
fi

LOG_DIR="$REPO_ROOT/experiments/$EXP_NAME"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_stdout.log"

echo "==============================================="
echo "  HAFI baseline training launch"
echo "==============================================="
echo "Config:  $CONFIG"
echo "Envs:    $N_ENVS  |  Device: $DEVICE  |  Seed: $SEED"
echo "Exp:     $EXP_NAME"
echo "Log:     $LOG_FILE"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  (no GPU)"
echo "==============================================="
echo ""

# nohup + tee → keeps running if user disconnects SSH, but logs visible live
nohup python -u train/train.py \
    --config "$CONFIG" \
    --n_envs "$N_ENVS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --exp_name "$EXP_NAME" \
    $TOTAL_ARG \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "Training started with PID $TRAIN_PID"
echo "$TRAIN_PID" > "$LOG_DIR/train.pid"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  # or wandb dashboard: https://wandb.ai/baijiaming46/affine_hafi"
echo ""
echo "Kill with:"
echo "  kill $TRAIN_PID  # or: kill \$(cat $LOG_DIR/train.pid)"
echo ""
echo "After training, evaluate with:"
echo "  python eval/eval.py --run experiments/$EXP_NAME --tiers L1,L2,L3"
