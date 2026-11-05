#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_HOME:?set HF_HOME to the Hugging Face cache folder}"
export HF_HOME
: "${RADHARMONY_OUTPUT_DIR:?set RADHARMONY_OUTPUT_DIR to the folder for caches and results}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
source "$SCRIPT_DIR/secrets.env"  # HF_TOKEN (gitignored)
cd "$SCRIPT_DIR/.."
.venv/bin/python eval_linear_probe_taix.py \
    --backbone medsiglip \
    --device cuda:0 \
    --batch_size 64 \
    --num_workers 8 \
    --n_seeds 5 \
    --n_bootstrap 200 \
    --n_train_samples 500 1000 2500 5000 10000 25000 \
    --output_dir "$RADHARMONY_OUTPUT_DIR/taix_lp_results_sweep"
