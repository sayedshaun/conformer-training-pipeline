# ASR Training Pipeline

Fine-tune a pretrained [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) FastConformer Hybrid (CTC + RNNT) speech recognition model on a new language, end to end: download a Common Voice–style corpus, train a tokenizer for the target language, fine-tune the model, and evaluate it.

Built around Mozilla's [Common Voice](https://commonvoice.mozilla.org/) data via the [Mozilla Data Collective](https://mozilladatacollective.com/) API, but the tokenizer/training/eval stages work with any manifest-based dataset in NeMo's JSON-lines format.

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
- A [Mozilla Data Collective](https://mozilladatacollective.com/) API key, unless you already have the corpus on disk (`skip_download: true`)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put your API key in a `.env` file at the project root (never in `config.yaml`):

```bash
echo "MDC_API_KEY=your-key-here" >> .env
```

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

Downloads the dataset archive (resumable, retried on stalled connections), extracts it, and converts each split's clips to 16kHz mono WAV, writing a NeMo-format manifest (`{split}_manifest.json`) per split. If you already have the corpus extracted locally, set `skip_download: true` in `config.yaml` and point `output_dir` at it.

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
prepare_data.py         CLI: data download + manifest generation
build_tokenizer.py      CLI: tokenizer training
train.py                CLI: model fine-tuning
eval.py                 CLI: model evaluation
src/
  config.py             YAML config-section loader
  dataset.py             Download, extraction, and manifest-building logic
  tokenizer.py            SentencePiece tokenizer training logic
  training.py             NeMo/Lightning fine-tuning logic
  evaluation.py           WER/CER evaluation logic
```
