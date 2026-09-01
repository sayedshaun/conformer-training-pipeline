<p align="center">
  <img src="assets/conformer_block.svg" alt="Conformer encoder block architecture" width="720">
  <br>
  <sub><a href="https://arxiv.org/abs/2005.08100">Conformer: Convolution-augmented Transformer for Speech Recognition</a></sub>
</p>

# Bengali Conformer Training Pipeline

Fine-tune a pretrained [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) Conformer/FastConformer Hybrid (CTC + RNNT) model for Bengali speech recognition — download data, train a tokenizer, fine-tune, evaluate.

Built around Mozilla's Bengali [Common Voice](https://commonvoice.mozilla.org/) release (via the [Mozilla Data Collective](https://mozilladatacollective.com/) API), [OpenSLR-53](https://www.openslr.org/53/) (Bengali), and Google's [FLEURS](https://huggingface.co/datasets/google/fleurs) `bn_in` split — though the tokenizer/train/eval stages work with any manifest-based dataset in NeMo's JSON-lines format, so the pipeline can be retargeted to another language by swapping the data sources and base checkpoint.

```
┌─────────────────┐     ┌────────────────────┐     ┌───────────┐     ┌───────────────┐
│ prepare_data.py │     │ build_tokenizer.py │     │ train.py  │     │ eval.py       │
│ download +      │  →  │ SentencePiece      │  →  │ fine-tune │  →  │ WER/CER on    │
│ build manifests │     │ BPE/unigram        │     │ CTC+RNNT  │     │ held-out test │
└─────────────────┘     └────────────────────┘     └───────────┘     └───────────────┘
```

Every stage is a thin CLI (`--config`, defaults to `config.yaml`) over core logic in [`src/`](src/). Each script reads only its own top-level section of the config, which is the single source of truth for all settings.

## Setup

```bash
git clone https://github.com/sayedshaun/conformer-training-pipeline.git
cd conformer-training-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "MDC_API_KEY=your-key-here" >> .env   # only needed for the mcv source
```

Requires Python 3.12 and an NVIDIA GPU. The `mcv` data source needs a [Mozilla Data Collective](https://mozilladatacollective.com/) API key unless `skip_download: true` is set; `openslr` and `fleurs` need no key. The `mcv` dataset also requires accepting its terms once on the [MDC dataset page](https://mozilladatacollective.com/datasets/cmqim44fo00tinr07mbu70eg7) before the API will serve a download.

Instead of a `.env` file, `MDC_API_KEY` can also be exported directly in the shell:

```bash
export MDC_API_KEY=your-key-here
```

## Training

Configure the run by editing [`config.yaml`](config.yaml) (data sources, tokenizer, model, training hyperparameters), then:

```bash
bash pipeline.sh
```

This installs `requirements.txt` into the active environment (assumes `python3`/`pip` and a CUDA-matched `torch` are already set up, e.g. inside the `venv` from Setup) and runs the pipeline stages in order:

```
prepare_data.py  →  data_stats.py  →  build_tokenizer.py  →  train.py
```

`pipeline.sh` takes no arguments — all configuration is read from `config.yaml`.

To resume an interrupted run, set `train.resume: true` and re-run — it picks up the latest checkpoint under `exp_dir/exp_name` (optimizer state, epoch, and step included), instead of starting over from `pretrained_model`. If no checkpoint exists yet, it falls back to a fresh run.

`eval.py` isn't part of `pipeline.sh` — run it separately once training finishes:

```bash
python eval.py
```

### Base models

`train.pretrained_model` accepts any checkpoint from NVIDIA's NeMo model registry, as long as its architecture matches `train.model_type` (defaults to `hybrid`):

- **`hybrid`** (default) — loads with `EncDecHybridRNNTCTCBPEModel`, requires a Hybrid Transducer+CTC checkpoint. Some options:

  | `pretrained_model` | Language |
  |---|---|
  | `stt_en_fastconformer_hybrid_large_pc` | English (default) |
  | `stt_multilingual_fastconformer_hybrid_large_pc` | Multilingual (be/de/en/es/fr/hr/it/pl/ru/ua) |

- **`ctc`** — loads with `EncDecCTCModelBPE`, for CTC-only checkpoints such as `stt_en_conformer_ctc_large` or `stt_en_fastconformer_ctc_large`.
- **`rnnt`** — loads with `EncDecRNNTBPEModel`, for Transducer-only checkpoints such as `stt_en_conformer_transducer_large`.

`eval.py`'s `decoder` setting only has an effect for `hybrid` checkpoints, since only they expose both a CTC and an RNNT head to switch between.

Browse the full, up-to-date list on the [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models) or [Hugging Face](https://huggingface.co/models?search=nvidia) — pick a checkpoint tagged Hybrid Transducer-CTC, CTC, or Transducer to match `model_type`, whose base language matches (or is close enough to bootstrap) the language you're fine-tuning for.

<details>
<summary><b>GPU / CUDA version</b></summary>

<br>

`requirements.txt` pins `torch` to a **CUDA 12.8** wheel and `numba-cuda` to `0.15.1`. NVIDIA drivers are backward-compatible, so any driver supporting CUDA 12.8+ (check with `nvidia-smi`) works unchanged. Only a driver **older** than 12.8 needs a swap:

```diff
- --extra-index-url https://download.pytorch.org/whl/cu128
+ --extra-index-url https://download.pytorch.org/whl/cu126
- numba-cuda[cu12]==0.15.1
+ numba-cuda[cu11]==0.15.1
- torch==2.11.0+cu128
+ torch==2.7.1+cu126
```

```bash
pip install --force-reinstall -r requirements.txt
python -c "import torch; print(torch.version.cuda)"
```

`torch` and `numba-cuda` must always target the same CUDA major version — they share the `cuda-bindings` dependency, pinned per-major-version by each, so mismatched majors fail dependency resolution.

`numba-cuda` is held at `0.15.1` specifically because later releases hit an `nvJitLink` linker error (`ERROR 4 in nvvmAddNVVMContainerToProgram`) inside NeMo's RNNT loss kernel. `0.15.1` only ships `cu11`/`cu12` extras — no `cu13`.

</details>

## Usage

Idempotent where it makes sense: `prepare_data.py` skips splits whose manifest already exists and resumes partial downloads. `train.py`/`build_tokenizer.py` always start fresh.

| Script | Config section | Purpose |
|---|---|---|
| `prepare_data.py` | `data:` | Dataset download + manifest generation |
| `build_tokenizer.py` | `tokenizer:` | SentencePiece tokenizer training |
| `train.py` | `train:` | Model fine-tuning |
| `eval.py` | `eval:` | WER/CER evaluation |

See [`config.yaml`](config.yaml) for the full set of keys and defaults.

<details>
<summary><b>1. Data preparation</b></summary>

<br>

`prepare_data.py` runs each source listed under `data.sources` and merges the results into one set of manifests:

- **`mcv`** ([`src/mcv.py`](src/mcv.py)) — downloads a Common Voice release via the Mozilla Data Collective API (resumable, retried on stalled connections) and builds `train`/`dev`/`test` manifests from Common Voice's own validated splits (`validated.tsv`, `dev.tsv`, `test.tsv`; invalidated/other clips are never used).
- **`openslr`** ([`src/openslr.py`](src/openslr.py)) — downloads the OpenSLR-53 Bengali corpus (16 zip shards) and builds manifests from every utterance in `utt_spk_text.tsv`. No official split exists, so a random `dev_utterances`/`test_utterances` sample (fixed seed) is held out and the rest becomes train.
- **`fleurs`** ([`src/fleurs.py`](src/fleurs.py)) — pulls all of `bn_in`'s splits from [google/fleurs](https://huggingface.co/datasets/google/fleurs) and folds them all into `train` (safe from contamination since dev/test already come from `mcv`/`openslr`).

Each source converts clips to 16kHz mono WAV under `output_dir/wavs/` and writes its own `{name}_{split}_manifest.json`; these get concatenated per-split into the final `train_manifest.json` / `dev_manifest.json` / `test_manifest.json`.

If a source's corpus is already extracted locally, set `skip_download: true` (globally or per source) and point `output_dir` at it. For `openslr`, `shards` can be trimmed to a subset (e.g. `["0", "1"]`) instead of `all` for a quick trial.

</details>

<details>
<summary><b>2. Tokenizer</b></summary>

<br>

Trains a SentencePiece tokenizer (BPE or unigram) over the given manifests' text and lays out the result the way NeMo's `ASRBPEMixin` expects (`tokenizer.model`, `tokenizer.vocab`, `vocab.txt`).

</details>

<details>
<summary><b>3. Training</b></summary>

<br>

Loads a pretrained hybrid (CTC + RNNT) checkpoint from NVIDIA's model registry via `train.pretrained_model`, swaps in the target-language tokenizer via `change_vocabulary`, and fine-tunes with PyTorch Lightning:

- Encoder freezing for the first N steps (`freeze_encoder_steps`), then automatic unfreezing
- Gradient accumulation and mixed precision
- Weights & Biases logging (`wandb_project`) and top-k checkpointing by validation WER, via NeMo's `exp_manager`

Final checkpoint: `experiments/<exp_name>/final.nemo`.

**Conformer vs. FastConformer** — `train.py` uses NeMo's generic `EncDecHybridRNNTCTCBPEModel`, so it isn't tied to FastConformer. To fine-tune a plain Conformer-Large model instead, just point `train.pretrained_model` in `config.yaml` at a Conformer hybrid RNNT-CTC checkpoint, e.g.:

```diff
- pretrained_model: stt_en_fastconformer_hybrid_large_pc
+ pretrained_model: stt_en_conformer_hybrid_large
```

No code changes needed. Note that Conformer-Large uses 4x subsampling (vs. FastConformer's 8x), so it processes more frames per second of audio and is more memory-hungry per batch — you may need to lower `train.batch_size` to avoid OOM.

</details>

<details>
<summary><b>4. Evaluation</b></summary>

<br>

Runs the fine-tuned model over a manifest, reports WER/CER, and optionally writes per-utterance predictions (`output_predictions`) for error analysis. Works with either decoding head (`decoder: ctc | rnnt`).

</details>

## License

[MIT](LICENSE)
