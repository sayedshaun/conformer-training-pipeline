#!/usr/bin/env bash
# Bootstrap script: clone the repo (or update it) and run the full pipeline
# sequentially: prepare_data.py -> data_stats.py -> build_tokenizer.py -> train.py
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
if [ ! -d "venv" ] && python3 -m venv venv 2>/dev/null; then
  :
elif [ -d "venv" ] && [ ! -f "venv/bin/pip" ]; then
  # venv exists but pip bootstrap failed (e.g. broken ensurepip on some
  # hosted notebook containers) - drop it and use the system python instead.
  rm -rf venv
fi
if [ -d "venv" ] && [ -f "venv/bin/pip" ]; then
  source venv/bin/activate
  PYTHON=python
fi

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
