"""CLI entrypoint for printing dataset statistics from the prepared manifests.

Reads config.yaml's `data:` section for `output_dir` and reports per-split
(and combined) counts, duration, and text-length stats for
train_manifest.json, dev_manifest.json, and test_manifest.json.
"""
import argparse
import json
import statistics
from pathlib import Path

from src.config import load_section

SPLITS = ("train", "dev", "test")


def load_manifest(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_stats(rows: list) -> dict:
    durations = [row["duration"] for row in rows]
    char_lens = [len(row["text"]) for row in rows]
    word_counts = [len(row["text"].split()) for row in rows]
    chars = set()
    for row in rows:
        chars.update(row["text"])

    return {
        "utterances": len(rows),
        "hours": sum(durations) / 3600,
        "duration_min": min(durations),
        "duration_max": max(durations),
        "duration_mean": statistics.mean(durations),
        "duration_median": statistics.median(durations),
        "chars_mean": statistics.mean(char_lens),
        "words_mean": statistics.mean(word_counts),
        "unique_chars": len(chars),
    }


def print_stats(name: str, stats: dict) -> None:
    print(f"\n{name}")
    print(f"  utterances     : {stats['utterances']:,}")
    print(f"  total duration : {stats['hours']:.2f} h")
    print(f"  duration (s)   : min={stats['duration_min']:.2f} "
          f"mean={stats['duration_mean']:.2f} "
          f"median={stats['duration_median']:.2f} "
          f"max={stats['duration_max']:.2f}")
    print(f"  text length    : {stats['chars_mean']:.1f} chars/utt, "
          f"{stats['words_mean']:.1f} words/utt")
    print(f"  unique chars   : {stats['unique_chars']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    cli_args = parser.parse_args()

    args = load_section(cli_args.config, "data", required=["output_dir"])
    output_dir = Path(args.output_dir)

    all_rows = []
    for split in SPLITS:
        manifest_path = output_dir / f"{split}_manifest.json"
        if not manifest_path.exists():
            print(f"\n{split}: {manifest_path} not found, skipping")
            continue
        rows = load_manifest(manifest_path)
        if not rows:
            print(f"\n{split}: {manifest_path} is empty, skipping")
            continue
        all_rows.extend(rows)
        print_stats(split, compute_stats(rows))

    if all_rows:
        print_stats("combined", compute_stats(all_rows))


if __name__ == "__main__":
    main()
