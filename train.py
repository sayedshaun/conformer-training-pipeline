"""CLI entrypoint for fine-tuning a pretrained NeMo FastConformer Hybrid model.

All settings come from config.yaml's `train:` section.
"""

import argparse
import os

# Must be set before torch is imported (via src.training). Allocator
# fragmentation alone can OOM a long run once utterance lengths vary.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dotenv import load_dotenv

from src.config import load_section
from src.training import run_training

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--batch", type=int, help="Overrides train.batch_size")
    parser.add_argument(
        "--gpus",
        help="GPU devices to train on: 'all', a count (e.g. 2), or a "
        "comma-separated list of indices (e.g. 0,1). Overrides train.devices.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a train: config value, e.g. --set lr=2e-5",
    )
    cli_args, _ = parser.parse_known_args()

    overrides = list(cli_args.set)
    if cli_args.batch is not None:
        overrides.append(f"batch_size={cli_args.batch}")
    if cli_args.gpus is not None:
        gpus = cli_args.gpus.strip().lower()
        if gpus == "all":
            overrides.append("devices=-1")
        elif "," in gpus:
            overrides.append(f"devices=[{gpus}]")
        else:
            overrides.append(f"devices={gpus}")

    args = load_section(
        cli_args.config,
        "train",
        required=["train_manifest", "val_manifest", "pretrained_model", "tokenizer_dir"],
        overrides=overrides,
    )
    run_training(args)


if __name__ == "__main__":
    main()
