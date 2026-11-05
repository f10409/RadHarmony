#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_HOME:?set HF_HOME to the Hugging Face cache folder}"
export HF_HOME
source "$SCRIPT_DIR/secrets.env"  # HF_TOKEN (gitignored)
export PYTHONUNBUFFERED=1
cd "$SCRIPT_DIR/.."
CUDA_VISIBLE_DEVICES=2 .venv/bin/python eval_linear_probe_taix.py --backbone medsiglip --device cuda:0 --batch_size 64 --num_workers 8 --n_seeds 5 --n_bootstrap 200
