#!/usr/bin/env bash
# Bootstrap script for a GPU Ubuntu server (bare metal, VM, or cloud
# instance): clone the repo (or update it) and run the full pipeline
# sequentially: prepare_data.py -> data_stats.py -> build_tokenizer.py -> train.py
#
# Assumes python3/pip and a CUDA-matched torch install are already set up
# (e.g. inside a venv you've activated) - this script does not create one.
#
# Usage (from anywhere):
#   curl -fsSL https://raw.githubusercontent.com/sayedshaun/conformer-training-pipeline/main/pipeline.sh | bash
set -euo pipefail

REPO_URL="https://github.com/sayedshaun/conformer-training-pipeline.git"
REPO_DIR="conformer-training-pipeline"

if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

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

"$PYTHON" prepare_data.py "$@"
"$PYTHON" data_stats.py
"$PYTHON" build_tokenizer.py
"$PYTHON" train.py "$@"
