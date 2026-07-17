#!/bin/bash
# FailureModeBench — ON-ROLF bootstrap: conda env + VLMEvalKit + sanity checks.
#
# PREREQUISITE (from your LAPTOP): bash scripts/sync_to_rolf.sh
# THEN on rolf:
#     cd ~/FailureModeBench
#     bash scripts/bootstrap_rolf.sh
set -e

USER=$(whoami)
SCRATCH=/local/scratch/$USER
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
echo "== project: $PROJ | user: $USER | scratch: $SCRATCH =="

# Keep the huge HF cache on scratch, not the quota-limited home.
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
mkdir -p "$HF_HOME"
echo "== HF_HOME=$HF_HOME =="

# 1. conda env ------------------------------------------------------------- #
if ! command -v conda >/dev/null 2>&1; then
    echo "== installing miniconda -> $SCRATCH/miniconda =="
    mkdir -p "$SCRATCH"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc_$USER.sh
    bash /tmp/mc_$USER.sh -b -p "$SCRATCH/miniconda"
    export PATH="$SCRATCH/miniconda/bin:$PATH"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^fmb "; then
    echo "== creating conda env 'fmb' (python 3.10) =="
    conda create -y -n fmb python=3.10
fi
conda activate fmb

# 2. deps ------------------------------------------------------------------ #
echo "== installing python deps =="
pip install -q -r requirements.txt
# VLMEvalKit for the VQA half (models + loaders + answer extraction)
if ! python -c "import vlmeval" 2>/dev/null; then
    echo "== installing VLMEvalKit =="
    pip install -q vlmeval || pip install -q git+https://github.com/open-compass/VLMEvalKit.git
fi

# 3. labelsets (build if missing) ----------------------------------------- #
if [ ! -f failuremodebench/labelsets/imagenet.json ]; then
    echo "== building labelsets =="
    python scripts/build_labelsets.py || echo "!! labelset build had failures; check output"
fi

# 4. sanity: GPU + a judge/key check -------------------------------------- #
python - <<'PY'
import torch
print(f"== torch {torch.__version__} cuda={torch.cuda.is_available()}"
      + (f" {torch.cuda.get_device_name(0)} "
         f"{torch.cuda.get_device_properties(0).total_memory//(1024**3)}GB"
         if torch.cuda.is_available() else " (CPU!)"))
import os
print("== ANTHROPIC_API_KEY set:", bool(os.environ.get("ANTHROPIC_API_KEY")))
from failuremodebench.config import DATASETS, MODELS
print(f"== registry: {len(DATASETS)} datasets, {len(MODELS)} models")
PY

echo
echo "SETUP COMPLETE. Set your judge key then run (inside screen):"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo "  screen -S fmb"
echo "  conda activate fmb && export HF_HOME=$HF_HOME"
echo "  CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh 2>&1 | tee run_rolf.log"
