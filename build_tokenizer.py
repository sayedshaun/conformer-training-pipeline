"""CLI entrypoint for training a SentencePiece tokenizer from NeMo manifests.

All settings come from config.yaml's `tokenizer:` section.
"""

import argparse

from src.config import load_section
from src.tokenizer import build_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    cli_args = parser.parse_args()

    args = load_section(
        cli_args.config,
        "tokenizer",
        required=["manifests", "tokenizer_dir", "vocab_size", "model_type"],
    )
    tokenizer_dir = build_tokenizer(
        args.manifests, args.tokenizer_dir, args.vocab_size, args.model_type
    )
    print(f"Trained tokenizer -> {tokenizer_dir}")


if __name__ == "__main__":
    main()
