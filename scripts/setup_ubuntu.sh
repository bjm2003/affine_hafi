#!/usr/bin/env bash
# Ubuntu training machine setup for affine_hafi.
#
# Usage on Ubuntu 20.04 (or 22.04) training machine:
#   cd ~
#   git clone git@github.com:bjm2003/affine_hafi.git
#   cd affine_hafi
#   bash scripts/setup_ubuntu.sh
#
# What this does:
#   1. Verifies conda is installed
#   2. Creates the affine_hafi conda env from environment.yml (idempotent)
#   3. Verifies CUDA is available (if GPU) OR falls back to CPU-only
#   4. Verifies MPC solver JIT compiles
#   5. Runs the fast pytest suite
#   6. Prompts for wandb login (unless WANDB_API_KEY env var is set)
#
# Exit codes:
#   0  everything ready
#   1  conda not installed
#   2  environment.yml missing
#   3  MPC verify failed
#   4  pytest failed

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

echo "==============================================="
echo "  affine_hafi Ubuntu training-machine setup"
echo "==============================================="
echo "Repo root: $REPO_ROOT"

# ---------- 1. Conda check ----------
if ! command -v conda >/dev/null 2>&1; then
    echo "[FATAL] conda not found. Install miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  source ~/.bashrc"
    exit 1
fi
echo "[1/7] conda found: $(conda --version)"

# ---------- 1b. Ensure libmamba solver (prevents OOM 'Killed' during solve) ----------
# conda's classic SAT solver can exhaust RAM (OOM-killed) on multi-channel envs.
# libmamba is C++-based, uses far less memory, and solves much faster.
echo "[2/7] Ensuring fast dependency solver (libmamba)..."
CONDA_BASE_EARLY=$(conda info --base)
if python -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('conda_libmamba_solver') else 1)" 2>/dev/null \
   || conda list -n base 2>/dev/null | grep -q "conda-libmamba-solver"; then
    echo "       libmamba solver already installed"
else
    echo "       Installing conda-libmamba-solver into base env..."
    conda install -n base -c conda-forge conda-libmamba-solver -y \
        || echo "       [WARN] libmamba install failed; will fall back to --solver flag / classic"
fi
# Set as default solver (harmless if already set); tolerate old conda without this key.
conda config --set solver libmamba 2>/dev/null \
    || conda config --set experimental_solver libmamba 2>/dev/null \
    || echo "       [WARN] could not persist solver setting; using per-command flag"

# ---------- 2. Environment.yml check ----------
if [[ ! -f "$REPO_ROOT/environment.yml" ]]; then
    echo "[FATAL] environment.yml not found in $REPO_ROOT"
    exit 2
fi

# ---------- 3. Detect GPU ----------
HAS_GPU=false
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        HAS_GPU=true
        echo "[3/7] GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/       /'
    fi
fi
if ! $HAS_GPU; then
    echo "[3/7] No GPU detected — will install CPU-only PyTorch"
    echo "       (If you have a GPU, install nvidia driver + CUDA 12.1 then re-run)"
fi

# ---------- 4. Create / update conda env ----------
# --solver libmamba is passed explicitly so this works even if the config
# write above was rejected by an older conda.
SOLVER_FLAG="--solver libmamba"
# Older conda (<23.9) doesn't accept --solver on env subcommands; probe once.
if ! conda env create --help 2>/dev/null | grep -q -- "--solver"; then
    SOLVER_FLAG=""
fi
if conda env list | grep -qE "^affine_hafi\s"; then
    echo "[4/7] Env 'affine_hafi' exists; updating from environment.yml..."
    conda env update -n affine_hafi -f environment.yml --prune $SOLVER_FLAG
else
    echo "[4/7] Creating env 'affine_hafi' from environment.yml..."
    if $HAS_GPU; then
        conda env create -f environment.yml $SOLVER_FLAG
    else
        # CPU-only variant: strip pytorch-cuda from env
        TMP_YML=$(mktemp)
        # Filter out the pytorch-cuda dependency line
        grep -v "pytorch-cuda" environment.yml > "$TMP_YML"
        conda env create -n affine_hafi -f "$TMP_YML" $SOLVER_FLAG
        rm -f "$TMP_YML"
        echo "       (Installed CPU-only PyTorch; edit environment.yml if you get GPU later)"
    fi
fi

# ---------- 5. Verify install: torch + MPC + pytest ----------
# Activate env (this shell only)
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate affine_hafi

echo "[5/7] Verifying PyTorch install..."
python -c "
import torch
print(f'  torch {torch.__version__}, cuda_available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device: {torch.cuda.get_device_name(0)}')
    print(f'  cuda: {torch.version.cuda}')
"

echo "[6/7] Verifying MPC solver..."
if ! python scripts/verify_mpc_offline.py; then
    echo "[FATAL] MPC solver verification failed"
    exit 3
fi

echo "[7/7] Running fast test suite..."
if ! pytest tests/ -v --tb=short -q; then
    echo "[FATAL] pytest failed"
    exit 4
fi

# ---------- 6. Wandb login ----------
echo ""
echo "==============================================="
echo "  All checks passed ✓"
echo "==============================================="

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    if ! [[ -f "$HOME/.netrc" ]] || ! grep -q "api.wandb.ai" "$HOME/.netrc"; then
        echo ""
        echo "wandb not logged in. Login now with:"
        echo "  wandb login"
        echo ""
        echo "Or set WANDB_API_KEY in your shell:"
        echo "  export WANDB_API_KEY=<your_key_from_wandb.ai/authorize>"
    else
        echo "wandb credentials already configured in ~/.netrc"
    fi
else
    echo "WANDB_API_KEY is set in environment"
fi

echo ""
echo "Next: kick off training with:"
echo "  bash scripts/launch_hafi_baseline.sh"
echo ""
