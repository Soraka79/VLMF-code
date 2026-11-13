#!/usr/bin/env bash

set -euo pipefail

CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
SCHEMAFLOW_ENV=/root/autodl-tmp/envs/schemaflow

if [[ ! -f "$CONDA_SH" ]]; then
  echo "missing Conda activation script: $CONDA_SH" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -x "$SCHEMAFLOW_ENV/bin/python" ]]; then
  echo "missing SchemaFlow environment: $SCHEMAFLOW_ENV" >&2
  return 1 2>/dev/null || exit 1
fi

source "$CONDA_SH"
conda activate "$SCHEMAFLOW_ENV"

export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export TMPDIR=/root/autodl-tmp/tmp
export PYTHONPYCACHEPREFIX=/root/autodl-tmp/.cache/pycache

mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$TORCH_HOME"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$TMPDIR"
mkdir -p "$PYTHONPYCACHEPREFIX"

echo "python=$(command -v python)"
echo "conda_prefix=${CONDA_PREFIX:-}"
echo "git_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
