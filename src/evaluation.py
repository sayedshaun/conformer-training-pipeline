"""Core evaluation logic for a fine-tuned NeMo ASR checkpoint."""
import json

import torch
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel


def load_manifest(path):
    audio_filepaths, references = [], []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            audio_filepaths.append(entry["audio_filepath"])
            references.append(entry["text"])
    return audio_filepaths, references


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASRModel.restore_from(args.model_path, map_location=device)
    model.eval()

    if hasattr(model, "change_decoding_strategy") and hasattr(model, "cur_decoder"):
        model.change_decoding_strategy(decoder_type=args.decoder)

    audio_filepaths, references = load_manifest(args.manifest)
    hypotheses = model.transcribe(audio_filepaths, batch_size=args.batch_size)
    if hasattr(hypotheses, "text"):
        hypotheses = hypotheses.text
    elif hypotheses and not isinstance(hypotheses[0], str):
        hypotheses = [h.text for h in hypotheses]

    wer = word_error_rate(hypotheses=hypotheses, references=references, use_cer=False)
    cer = word_error_rate(hypotheses=hypotheses, references=references, use_cer=True)
    print(f"Utterances: {len(references)}")
    print(f"WER: {wer * 100:.2f}%")
    print(f"CER: {cer * 100:.2f}%")

    if args.output_predictions:
        with open(args.output_predictions, "w") as f:
            f.writelines(
                json.dumps({"audio_filepath": path, "reference": ref, "prediction": hyp}) + "\n"
                for path, ref, hyp in zip(audio_filepaths, references, hypotheses)
            )
        print(f"Predictions written to {args.output_predictions}")

    return wer, cer
