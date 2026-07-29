"""Trains a SentencePiece tokenizer from NeMo manifest text and lays it out the
way NeMo's ASRBPEMixin expects (tokenizer.model, tokenizer.vocab, vocab.txt)."""
import json
from pathlib import Path

import sentencepiece as spm


def _read_manifest_text(manifest_paths: list) -> list:
    lines = []
    for manifest_path in manifest_paths:
        with open(manifest_path) as f:
            for line in f:
                lines.append(json.loads(line)["text"])
    return lines


def build_tokenizer(manifest_paths: list, tokenizer_dir: str, vocab_size: int, model_type: str) -> Path:
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = tokenizer_dir / "corpus.txt"
    corpus_path.write_text("\n".join(_read_manifest_text(manifest_paths)))

    model_prefix = tokenizer_dir / "tokenizer"
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,  # keep every Bengali grapheme/conjunct, not just the top ~99.95%
        bos_id=-1,
        eos_id=-1,
        pad_id=0,
        unk_id=1,
        max_sentence_length=4096,
    )
    corpus_path.unlink()

    # NeMo also wants a plain-text vocab.txt alongside tokenizer.model/.vocab
    vocab_lines = [line.split("\t")[0] for line in (tokenizer_dir / "tokenizer.vocab").read_text().splitlines()]
    (tokenizer_dir / "vocab.txt").write_text("\n".join(vocab_lines) + "\n")

    return tokenizer_dir
