"""CLI entrypoint for normalizing transcripts in the prepared manifests.

Rewrites train/dev/test_manifest.json in place (keeping a .bak) so the text
is consistent before `build_tokenizer.py` sees it. Three passes:

  1. NFC-normalize, which unifies the two spellings of the nukta letters
     (ড় ঢ় য়) onto base+nukta - they are Unicode composition exclusions, so
     NFC decomposes the precomposed codepoints rather than the reverse.
     Without this the tokenizer treats the two spellings of one grapheme as
     unrelated, and eight sources will not agree on which they use.
  2. Strip characters that are not pronounced (punctuation, Devanagari danda,
     zero-width joiners and bidi marks).
  3. Drop utterances whose text still contains Latin letters or digits.
     Stripping those instead would leave a transcript that no longer matches
     the audio, which is worse than losing the utterance.

Assamese ra/wa are mapped onto their Bengali equivalents; they appear only via
cross-language contamination in the multilingual corpora.
"""
import argparse
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from src.config import load_section

SPLITS = ("train", "dev", "test")

# Not pronounced - safe to remove without invalidating the audio pairing.
STRIP_CHARS = "‘’“”‚„…–—:;/%[](){}<>°+*=_~|@#$£¥€॥।\"'`^\\&!?.,"
INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\ufeff]")
ASSAMESE_MAP = {"\u09f0": "\u09b0", "\u09f1": "\u09ac"}  # Assamese ra -> ra, wa -> ba
HAS_LATIN_OR_DIGIT = re.compile(r"[A-Za-zÀ-ɏ0-9]")
STRIP_TABLE = {ord(c): " " for c in STRIP_CHARS}
STRIP_TABLE.update({ord(k): v for k, v in ASSAMESE_MAP.items()})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE.sub("", text)
    text = text.translate(STRIP_TABLE)
    return " ".join(text.split())


def process(path: Path, dry_run: bool) -> dict:
    kept, dropped_empty, dropped_latin, changed = [], 0, 0, 0
    dropped_chars = Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            original = row["text"]
            text = normalize(original)
            if not text:
                dropped_empty += 1
                continue
            if HAS_LATIN_OR_DIGIT.search(text):
                dropped_latin += 1
                dropped_chars.update(HAS_LATIN_OR_DIGIT.findall(text))
                continue
            if text != original:
                changed += 1
            row["text"] = text
            kept.append(row)

    if not dry_run:
        shutil.copy2(path, path.with_suffix(".json.bak"))
        with open(path, "w") as out:
            out.writelines(
                json.dumps(r, ensure_ascii=False) + "\n" for r in kept
            )

    alphabet = set()
    for row in kept:
        alphabet.update(row["text"])
    return {
        "kept": len(kept),
        "changed": changed,
        "dropped_empty": dropped_empty,
        "dropped_latin": dropped_latin,
        "unique_chars": len(alphabet),
        "top_dropped": dropped_chars.most_common(8),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report only; leave manifests untouched"
    )
    cli_args = parser.parse_args()

    args = load_section(cli_args.config, "data", required=["output_dir"])
    output_dir = Path(args.output_dir)

    for split in SPLITS:
        path = output_dir / f"{split}_manifest.json"
        if not path.exists():
            print(f"\n{split}: {path} not found, skipping")
            continue
        s = process(path, cli_args.dry_run)
        total = s["kept"] + s["dropped_empty"] + s["dropped_latin"]
        print(f"\n{split}  ({path})")
        print(f"  kept           : {s['kept']:,} of {total:,}")
        print(f"  text rewritten : {s['changed']:,}")
        print(f"  dropped (empty): {s['dropped_empty']:,}")
        print(f"  dropped (latin/digit): {s['dropped_latin']:,}")
        print(f"  unique chars   : {s['unique_chars']}")
        if s["top_dropped"]:
            print(f"  most common offenders: {s['top_dropped']}")

    if cli_args.dry_run:
        print("\n(dry run - no files written)")
    else:
        print("\nOriginals saved alongside as *_manifest.json.bak")


if __name__ == "__main__":
    main()
