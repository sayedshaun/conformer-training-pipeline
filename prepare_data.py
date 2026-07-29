"""CLI entrypoint for downloading a Common Voice release and building NeMo manifests.

All settings come from config.yaml's `data:` section. The API key comes from
the MDC_API_KEY env var (.env), never from config.yaml.
"""
import argparse
import os

from dotenv import load_dotenv

from src.config import load_section
from src.dataset import prepare_dataset

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    cli_args = parser.parse_args()

    args = load_section(cli_args.config, "data", required=["dataset_id", "output_dir"])
    args.api_key = os.environ.get("MDC_API_KEY")

    if not args.api_key and not args.skip_download:
        raise SystemExit("Set MDC_API_KEY in .env, or skip_download: true in config.yaml")

    prepare_dataset(args)


if __name__ == "__main__":
    main()
