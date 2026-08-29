#!/usr/bin/env bash
# Bootstrap script for Google Colab notebooks: clone the repo (or update it)
# and run the full pipeline sequentially: prepare_data.py -> data_stats.py ->
# build_tokenizer.py -> train.py
#
# Colab containers ship a preinstalled torch/CUDA stack and often have a
# broken venv/ensurepip, so this installs straight into the system
# environment with --force-reinstall to override whatever Colab preinstalled
# rather than layering on top of it.
#
# Usage (from a Colab notebook cell):
#   !curl -fsSL https://raw.githubusercontent.com/sayedshaun/conformer-training-pipeline/main/pipeline_colab.sh | bash
#
# To override config.yaml values (forwarded to prepare_data.py/train.py),
# pass them after `-s --`:
#   !curl -fsSL .../pipeline_colab.sh | bash -s -- --dataset fleurs --batch 16
set -euo pipefail

REPO_URL="https://github.com/sayedshaun/conformer-training-pipeline.git"
REPO_DIR="conformer-training-pipeline"

if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# Trim pip's noisy resolver/build output down to just package names and
# download progress, without silencing errors.
pip_install() {
  set +o pipefail
  python3 -m pip install --disable-pip-version-check "$@" 2>&1 \
    | grep --line-buffered -E '^(Collecting|Downloading|Installing collected packages|Successfully installed|ERROR)'
  local status=${PIPESTATUS[0]}
  set -o pipefail
  return "$status"
}

pip_install --force-reinstall -r requirements.txt

python3 prepare_data.py "$@"
python3 data_stats.py
python3 build_tokenizer.py
python3 train.py "$@"
