# ASR Training Pipeline

Fine-tune a pretrained [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) FastConformer Hybrid (CTC + RNNT) speech recognition model on a new language, end to end: download a Common Voice–style corpus, train a tokenizer for the target language, fine-tune the model, and evaluate it.

Built around Mozilla's [Common Voice](https://commonvoice.mozilla.org/) data via the [Mozilla Data Collective](https://mozilladatacollective.com/) API, the [OpenSLR-53](https://www.openslr.org/53/) Bengali ASR corpus, and Google's [FLEURS](https://huggingface.co/datasets/google/fleurs) benchmark, but the tokenizer/training/eval stages work with any manifest-based dataset in NeMo's JSON-lines format.

## Pipeline overview

```
prepare_data.py   →  build_tokenizer.py  →  train.py           →  eval.py
(download + build     (SentencePiece BPE/    (fine-tune FastConformer  (WER/CER on
 NeMo manifests)       unigram tokenizer)      Hybrid CTC+RNNT)          held-out test set)
```

Every stage is a thin CLI wrapper (`argparse`, one `--config` flag) around core logic in [`src/`](src/). All settings live in one place — [`config.yaml`](config.yaml) — with each script reading only its own top-level section.

## Requirements

- Python 3.12
- An NVIDIA GPU (training and evaluation both assume CUDA)
- A [Mozilla Data Collective](https://mozilladatacollective.com/) API key for the `mcv` data source, unless you already have the corpus on disk (`skip_download: true`). The `openslr` and `fleurs` sources need no API key — `openslr` downloads directly from OpenSLR mirrors, and `fleurs` pulls via the HuggingFace `datasets` library.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put your API key in a `.env` file at the project root (never in `config.yaml`):

```bash
echo "MDC_API_KEY=your-key-here" >> .env
```

### GPU / CUDA version

`requirements.txt` pins `torch` to a wheel built for **CUDA 12.8** and `numba-cuda` to `0.15.1`, matching the driver on the reference training server. pip does not auto-detect your driver's CUDA version, so on a machine with a different one you must swap these pins yourself — check your driver's max supported CUDA version with `nvidia-smi` (top-right of the header) first.

NVIDIA drivers are backward-compatible with older CUDA builds, so a driver that supports CUDA 13 can still run these CUDA-12.8 wheels without any changes — only a driver **older** than 12.8 requires switching to an older CUDA build:

```diff
- --extra-index-url https://download.pytorch.org/whl/cu128
+ --extra-index-url https://download.pytorch.org/whl/cu126
- numba-cuda[cu12]==0.15.1
+ numba-cuda[cu11]==0.15.1
- torch==2.11.0+cu128
+ torch==2.7.1+cu126
```

Then reinstall and verify:

```bash
pip install --force-reinstall -r requirements.txt
python -c "import torch; print(torch.version.cuda)"
```

The `torch` CUDA build and the `numba-cuda` extra must always target the same CUDA major version, since `cuda-bindings` (a shared dependency) is pinned per-major-version by each — mixing e.g. a `cu128` torch build with `numba-cuda[cu13]` fails dependency resolution (`cuda-bindings<13` vs `==13.*`).

`numba-cuda` is deliberately held at `0.15.1` rather than a newer release: versions after `0.15.1` hit an `nvJitLink` linker error (`ERROR 4 in nvvmAddNVVMContainerToProgram`) inside NeMo's RNNT loss CUDA kernel. `0.15.1` only ships `cu11`/`cu12` extras (no `cu13`), so this pin is CUDA-12-and-older only — check the `numba-cuda` PyPI page before assuming a `cu13` extra exists on whatever version you land on.

## Quick start

Run the full pipeline (clone/update repo, install deps, prep data, print stats, train) on a fresh machine with one command:

```bash
curl -fsSL https://raw.githubusercontent.com/sayedshaun/conformer-training-pipeline/main/pipeline.sh | bash
```

This runs [`pipeline.sh`](pipeline.sh), which does, in order: `git clone`/`git pull` → `pip install -r requirements.txt` → `python prepare_data.py` → `python data_stats.py` → `python train.py`.

## Configuration

Every script takes only `-c/--config` (defaults to `config.yaml`) and reads its own section:

| Script | Config section | Purpose |
|---|---|---|
| `prepare_data.py` | `data:` | Dataset download + manifest generation |
| `build_tokenizer.py` | `tokenizer:` | SentencePiece tokenizer training |
| `train.py` | `train:` | Model fine-tuning |
| `eval.py` | `eval:` | WER/CER evaluation |

See [`config.yaml`](config.yaml) for the full set of keys and their defaults — it's intentionally kept as the single source of truth rather than duplicated here.

## Usage

Run each stage in order:

```bash
# 1. Download the corpus and build train/dev/test manifests
python prepare_data.py

# 2. Train a tokenizer on the training manifest
python build_tokenizer.py

# 3. Fine-tune the pretrained model
python train.py

# 4. Evaluate the fine-tuned checkpoint
python eval.py
```

Each stage is idempotent where it makes sense — `prepare_data.py` skips splits whose manifest already exists and resumes partial downloads; re-running `train.py` or `build_tokenizer.py` starts a fresh run.

### 1. Data preparation

`prepare_data.py` pulls one or more data sources, listed under `data.sources` in `config.yaml`, and merges them into a single set of manifests:

- **`mcv`** ([`src/mcv.py`](src/mcv.py)) — downloads a Common Voice release via the Mozilla Data Collective API (resumable, retried on stalled connections), extracts it, and builds `train`/`dev`/`test` manifests from Common Voice's own validated splits (`validated.tsv`, `dev.tsv`, `test.tsv` — invalidated/other clips are never used).
- **`openslr`** ([`src/openslr.py`](src/openslr.py)) — downloads the [OpenSLR-53](https://www.openslr.org/53/) Bengali corpus (16 zip shards), extracts them, and builds manifests from every utterance in `utt_spk_text.tsv`. The corpus has no official split, so a random `dev_utterances`/`test_utterances` sample (fixed seed) is held out and the rest becomes train.
- **`fleurs`** ([`src/fleurs.py`](src/fleurs.py)) — pulls all of `bn_in`'s splits (train/validation/test) from [google/fleurs](https://huggingface.co/datasets/google/fleurs) via the HuggingFace `datasets` library and folds them all into our `train` manifest. Since our own dev/test held-out sets already come from the `mcv`/`openslr` sources, there's no eval-contamination risk in also training on fleurs' own dev/test.

Each source converts its clips to 16kHz mono WAV under `output_dir/wavs/` and writes its own `{name}_{split}_manifest.json`. `prepare_data.py` then concatenates same-split files across sources into the final `train_manifest.json`, `dev_manifest.json`, `test_manifest.json` that `build_tokenizer.py`/`train.py`/`eval.py` read.

If you already have a source's corpus extracted locally, set `skip_download: true` (globally, or per source) in `config.yaml` and point `output_dir` at it. For `openslr`, `shards` can be trimmed to a subset (e.g. `["0", "1"]`) instead of `all` for a smaller trial run.

### 2. Tokenizer

Trains a SentencePiece tokenizer (BPE or unigram) over the text in the given manifests, and lays the output out the way NeMo's `ASRBPEMixin` expects (`tokenizer.model`, `tokenizer.vocab`, `vocab.txt`).

### 3. Training

Loads a pretrained FastConformer Hybrid checkpoint from NVIDIA's model registry, swaps in the target-language tokenizer via `change_vocabulary`, and fine-tunes it with PyTorch Lightning. Supports:

- Encoder freezing for the first N steps (`freeze_encoder_steps`), then automatic unfreezing
- Gradient accumulation and mixed precision
- Weights & Biases logging (`wandb_project`) and top-k checkpointing by validation WER, via NeMo's `exp_manager`

The final checkpoint is saved to `experiments/<exp_name>/final.nemo`.

### 4. Evaluation

Runs the fine-tuned model over a manifest, reports WER and CER, and optionally writes per-utterance predictions to a JSON-lines file (`output_predictions`) for error analysis. Works with either the CTC or RNNT decoding head (`decoder: ctc | rnnt`).

## Project layout

```
config.yaml            Single source of truth for all pipeline settings
prepare_data.py         CLI: runs each configured data source, merges manifests
build_tokenizer.py      CLI: tokenizer training
train.py                CLI: model fine-tuning
eval.py                 CLI: model evaluation
src/
  config.py             YAML config-section loader
  mcv.py                 Common Voice / Mozilla Data Collective source
  openslr.py             OpenSLR-53 Bengali corpus source
  fleurs.py             google/fleurs (bn_in, all splits folded into train) source
  download.py             Shared resumable-download helper
  audio.py                Shared clip-to-16kHz-mono-WAV conversion helper
  tokenizer.py            SentencePiece tokenizer training logic
  training.py             NeMo/Lightning fine-tuning logic
  evaluation.py           WER/CER evaluation logic
```
