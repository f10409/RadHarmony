#!/usr/bin/env bash
set -e

# Run from the repo root so `uv pip install -e .` finds pyproject.toml and the
# venvs are created there, no matter which directory the notebook launches from.
cd "$(dirname "$0")/.."

# The `notebook` extra pulls ipywidgets so tqdm progress bars render in Jupyter
# (otherwise you get an "IProgress not found" warning).
declare -A envs=(
  ["medgemma"]="medsiglip,medgemma_gen,notebook"
  ["chexagent"]="chexagent_gen,notebook"
  ["maira2"]="maira2_gen,notebook"
)

# CUDA build for the whole PyTorch stack. The default PyPI wheels are cu126,
# which only support up to sm_90 (Hopper) and CANNOT run on Blackwell GPUs
# (sm_120, e.g. RTX PRO 6000). cu128 supports sm_75..sm_120. Override with
# TORCH_BACKEND=auto (uv auto-detects the driver) or another cuXXX / cpu tag.
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"

for name in "${!envs[@]}"; do
  venv_dir=".venv-${name}"
  py="${venv_dir}/bin/python"
  extras="${envs[$name]}"

  echo "=== ${name}: creating venv ==="
  uv venv "${venv_dir}" --python 3.11

  echo "=== ${name}: installing extras [${extras}] (torch backend: ${TORCH_BACKEND}) ==="
  uv pip install -e ".[${extras}]" --python "${py}" --torch-backend="${TORCH_BACKEND}"

  if [[ "${name}" == "medgemma" ]]; then
    echo "=== ${name}: installing extras [radeval] ==="
    uv pip install -e ".[radeval]" --python "${py}" --torch-backend="${TORCH_BACKEND}"
  fi

  echo "=== ${name}: installing ipykernel ==="
  uv pip install ipykernel --python "${py}"

  echo "=== ${name}: registering kernel ==="
  "${py}" -m ipykernel install --user --name "${name}" --display-name "RadHarmony-(${name})"
done

echo "Done. Registered kernels:"
jupyter kernelspec list