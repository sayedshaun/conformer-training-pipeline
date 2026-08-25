#!/usr/bin/env bash
# Bootstrap script for Google Colab notebooks: clone the repo (or update it)
# and run the full pipeline sequentially: prepare_data.py -> data_stats.py ->
# train.py
#
# Colab containers ship a preinstalled torch/CUDA stack and often have a
# broken venv/ensurepip, so this installs straight into the system
# environment with --force-reinstall to override whatever Colab preinstalled
# rather than layering on top of it.
#
# Usage (from a Colab notebook cell):
#   !curl -fsSL https://raw.githubusercontent.com/sayedshaun/conformer-training-pipeline/main/pipeline_colab.sh | bash
set -euo pipefail

REPO_URL="https://github.com/sayedshaun/conformer-training-pipeline.git"
REPO_DIR="conformer-training-pipeline"

if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

python3 -m pip install --force-reinstall -r requirements.txt

python3 prepare_data.py
python3 data_stats.py
python3 train.py
