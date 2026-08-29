#!/usr/bin/env bash
# Bootstrap script for hosted notebooks (Kaggle / Colab): clone the repo (or
# update it) and run the full pipeline sequentially:
#   prepare_data.py -> data_stats.py -> build_tokenizer.py -> train.py
#
# Differences from pipeline.sh:
#   - No venv: notebooks already ship a GPU-matched Python, and creating a
#     venv risks reinstalling a CPU-only or CUDA-mismatched torch.
#   - Logs in to W&B automatically from WANDB_API_KEY, or from Kaggle's
#     "Add-ons > Secrets" if WANDB_API_KEY isn't set but kaggle_secrets is
#     available.
#
# Usage (from a notebook cell):
#   !WANDB_API_KEY=xxxx curl -fsSL \
#     https://raw.githubusercontent.com/sayedshaun/conformer-training-pipeline/main/notebook.sh \
#     | bash -s -- --your-arg value
#
# Or, since the curl-pipe form makes `-s -- ...` awkward to compose, pass the
# same flags via an ARGS env var instead:
#   !ARGS="--batch 16 --dataset fleurs --gpus all" curl -fsSL .../notebook.sh | bash
set -euo pipefail

REPO_URL="https://github.com/sayedshaun/conformer-training-pipeline.git"
REPO_DIR="conformer-training-pipeline"

if [ $# -eq 0 ] && [ -n "${ARGS:-}" ]; then
  eval "set -- $ARGS"
fi

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

pip_install --force-reinstall -r requirements.txt

if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY="$("$PYTHON" -c '
try:
    from kaggle_secrets import UserSecretsClient
    print(UserSecretsClient().get_secret("WANDB_API_KEY"))
except Exception:
    pass
' 2>/dev/null)"
  export WANDB_API_KEY
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  "$PYTHON" -m wandb login "$WANDB_API_KEY"
else
  echo "WANDB_API_KEY not set (env var or Kaggle secret) - skipping wandb login." >&2
fi

"$PYTHON" prepare_data.py "$@"
"$PYTHON" data_stats.py
"$PYTHON" build_tokenizer.py
"$PYTHON" train.py "$@"
