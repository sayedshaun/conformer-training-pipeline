#!/usr/bin/env bash
# Runs the full pipeline sequentially from within an already-cloned repo:
# prepare_data.py -> data_stats.py -> build_tokenizer.py -> train.py
#
# Assumes python3/pip and a CUDA-matched torch install are already set up
# (e.g. inside a venv you've activated) - this script does not create one.
#
# Usage (from the repo root):
#   ./pipeline.sh
set -euo pipefail

PYTHON=python3

# Trim pip's noisy resolver/build output down to just package names and
# download progress, without silencing errors.
pip_install() {
  set +o pipefail
  "$PYTHON" -m pip install --disable-pip-version-check "$@" 2>&1 \
    | grep --line-buffered -E '^(Collecting|Downloading|Installing collected packages|Successfully installed|ERROR)'
  local status=${PIPESTATUS[0]}
  set -o pipefail
  return "$status"
}

pip_install -r requirements.txt

"$PYTHON" prepare_data.py
"$PYTHON" data_stats.py
"$PYTHON" build_tokenizer.py
"$PYTHON" train.py
