#!/usr/bin/env bash
set -e

declare -A envs=(
  ["medgemma"]="medsiglip,medgemma_gen"
  ["chexagent"]="chexagent_gen"
  ["maira2"]="maira2_gen"
)

for name in "${!envs[@]}"; do
  venv_dir=".venv-${name}"
  py="${venv_dir}/bin/python"
  extras="${envs[$name]}"

  echo "=== ${name}: creating venv ==="
  uv venv "${venv_dir}" --python 3.11

  echo "=== ${name}: installing extras [${extras}] ==="
  uv pip install -e ".[${extras}]" --python "${py}"

  if [[ "${name}" == "medgemma" ]]; then
    echo "=== ${name}: installing extras [radeval] ==="
    uv pip install -e ".[radeval]" --python "${py}"
  fi

  echo "=== ${name}: installing ipykernel ==="
  uv pip install ipykernel --python "${py}"

  echo "=== ${name}: registering kernel ==="
  "${py}" -m ipykernel install --user --name "${name}" --display-name "RadHarmony-(${name})"
done

echo "Done. Registered kernels:"
jupyter kernelspec list