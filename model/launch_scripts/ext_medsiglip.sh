#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${EMORY_JOINED_PARQUET:?set EMORY_JOINED_PARQUET to the emory_joined.parquet file}"
: "${EMORY_ZARR_DIR:?set EMORY_ZARR_DIR to the folder of the Emory zarr images}"
: "${MIMIC_TAIX_LABELS:?set MIMIC_TAIX_LABELS to the mimic_taix_labels.parquet file}"
: "${MIMIC_DICOM_DIR:?set MIMIC_DICOM_DIR to the MIMIC-CXR DICOM folder with temp_mimic.csv}"
export EMORY_JOINED_PARQUET EMORY_ZARR_DIR MIMIC_TAIX_LABELS MIMIC_DICOM_DIR
export PYTHONUNBUFFERED=1
: "${HF_HOME:?set HF_HOME to the Hugging Face cache folder}"
export HF_HOME
source "$SCRIPT_DIR/secrets.env"  # HF_TOKEN (gitignored)
export CUDA_VISIBLE_DEVICES=2
cd "$SCRIPT_DIR/.."

echo "=== medsiglip → emory ==="
.venv/bin/python eval_linear_probe_external.py \
    --backbone medsiglip \
    --target emory \
    --device cuda:0 --batch_size 64 --num_workers 4 \
    --n_seeds 5 --n_bootstrap 200 --max_samples 50000

echo "=== medsiglip → mimic ==="
.venv/bin/python eval_linear_probe_external.py \
    --backbone medsiglip \
    --target mimic \
    --device cuda:0 --batch_size 64 --num_workers 4 \
    --n_seeds 5 --n_bootstrap 200
