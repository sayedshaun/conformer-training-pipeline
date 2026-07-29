"""CLI entrypoint for fine-tuning a pretrained NeMo FastConformer Hybrid model.

All settings come from config.yaml's `train:` section.
"""

import argparse
import os

# Must be set before torch is imported (via src.training) - this GPU is small
# enough (< 4 GiB) that allocator fragmentation alone can OOM a long run.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dotenv import load_dotenv

from src.config import load_section
from src.training import run_training

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    cli_args = parser.parse_args()

    args = load_section(
        cli_args.config,
        "train",
        required=["train_manifest", "val_manifest", "pretrained_model", "tokenizer_dir"],
    )
    run_training(args)


if __name__ == "__main__":
    main()
