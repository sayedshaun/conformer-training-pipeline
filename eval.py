"""CLI entrypoint for evaluating a fine-tuned NeMo ASR model, reporting WER and CER.

All settings come from config.yaml's `eval:` section.
"""
import argparse

from src.config import load_section
from src.evaluation import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    cli_args = parser.parse_args()

    args = load_section(cli_args.config, "eval", required=["model_path", "manifest"])
    run_eval(args)


if __name__ == "__main__":
    main()
