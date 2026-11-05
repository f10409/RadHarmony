#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LEJEPA_CKPT:?set LEJEPA_CKPT to the LeJEPA checkpoint file (mimic-cxp-from-imagenet-dp01-bs160-e10_last.ckpt)}"
: "${HF_HOME:?set HF_HOME to the Hugging Face cache folder}"
export HF_HOME
export PYTHONUNBUFFERED=1
cd "$SCRIPT_DIR/.."
CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_linear_probe_taix.py --backbone lejepa --ckpt "$LEJEPA_CKPT" --device cuda:0 --batch_size 64 --num_workers 8 --n_seeds 5 --n_bootstrap 200
