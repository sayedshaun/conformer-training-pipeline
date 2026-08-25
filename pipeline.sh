#!/usr/bin/env bash
# Bootstrap script: clone the repo (or update it) and run the full pipeline
# sequentially: prepare_data.py -> data_stats.py -> train.py
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

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

python prepare_data.py
python data_stats.py
python train.py
