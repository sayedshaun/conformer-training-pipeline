"""Core dataset logic: download Mozilla Data Collective releases and build NeMo manifests."""
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000
DOWNLOAD_READ_TIMEOUT = 60  # seconds of silence on the socket before treating it as stalled
DOWNLOAD_MAX_RETRIES = 8


def download_dataset(dataset_id: str, api_key: str, dest: Path) -> Path:
    resp = requests.post(
        f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    download_url = resp.json()["downloadUrl"]

    archive_path = dest / f"{dataset_id}.tar.gz"
    dest.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        resume_pos = archive_path.stat().st_size if archive_path.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
        try:
            with requests.get(
                download_url, stream=True, headers=headers, timeout=(10, DOWNLOAD_READ_TIMEOUT),
            ) as r:
                if resume_pos and r.status_code == 416:
                    # Our local partial file doesn't line up with what the server has (e.g. the
                    # presigned URL rotated mid-download). Check the real size: if we already
                    # have it all, we're done; otherwise the partial file is stale, discard it.
                    head = requests.head(download_url, timeout=30)
                    true_size = int(head.headers.get("content-length", 0))
                    if true_size and resume_pos >= true_size:
                        print(f"{archive_path} is already fully downloaded ({resume_pos} bytes)")
                        return archive_path
                    print("Local partial file doesn't match the remote object, restarting download")
                    archive_path.unlink()
                    continue
                if resume_pos and r.status_code == 200:
                    # Server ignored the Range header, so it's sending the file from scratch.
                    resume_pos = 0
                r.raise_for_status()
                total = resume_pos + int(r.headers.get("content-length", 0))
                mode = "ab" if resume_pos else "wb"
                with open(archive_path, mode) as f, tqdm(
                    total=total, initial=resume_pos, unit="B", unit_scale=True, desc="download",
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            return archive_path
        except requests.exceptions.RequestException as e:
            print(f"Download stalled or failed (attempt {attempt}/{DOWNLOAD_MAX_RETRIES}): {e}")
            if attempt == DOWNLOAD_MAX_RETRIES:
                raise
            print("Retrying, resuming from the last downloaded byte...")

    raise RuntimeError("unreachable")


def find_corpus_root(path: Path):
    """Common Voice archives nest the actual corpus (clips/ + *.tsv) inside a
    locale subdirectory, e.g. cv-corpus-26.0-.../bn/clips - search for it."""
    if (path / "clips").is_dir():
        return path
    for p in path.iterdir():
        if p.is_dir():
            found = find_corpus_root(p)
            if found:
                return found
    return None


def extract_dataset(archive_path: Path, dest: Path) -> Path:
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest)
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    top = subdirs[0] if len(subdirs) == 1 else dest
    corpus_dir = find_corpus_root(top)
    if corpus_dir is None:
        raise RuntimeError(f"Could not find a clips/ directory anywhere under {top}")
    return corpus_dir


def find_existing_corpus(output_dir: Path):
    if not output_dir.exists():
        return None
    for p in output_dir.iterdir():
        if p.is_dir():
            found = find_corpus_root(p)
            if found:
                return found
    return None


def convert_clip(args):
    src_path, dst_path = args
    if not dst_path.exists():
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
        audio.export(dst_path, format="wav")
    with sf.SoundFile(dst_path) as f:
        return len(f) / f.samplerate


def build_manifest(corpus_dir: Path, split: str, clips_out_dir: Path, workers: int) -> int:
    if split == "train":
        # train.tsv is only the official CV subset of validated.tsv; pull in the
        # rest of the validated pool too, excluding whatever dev/test hold out.
        df = pd.read_csv(corpus_dir / "validated.tsv", sep="\t")
        held_out = set()
        for other_split in ("dev", "test"):
            other_tsv = corpus_dir / f"{other_split}.tsv"
            if other_tsv.exists():
                held_out.update(pd.read_csv(other_tsv, sep="\t")["path"])
        df = df[~df["path"].isin(held_out)]
    else:
        df = pd.read_csv(corpus_dir / f"{split}.tsv", sep="\t")
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    jobs, rows = [], []
    for _, row in df.iterrows():
        src = corpus_dir / "clips" / row["path"]
        dst = clips_out_dir / (Path(row["path"]).stem + ".wav")
        jobs.append((src, dst))
        rows.append((dst, str(row["sentence"]).strip()))

    manifest_path = clips_out_dir.parent / f"{split}_manifest.json"
    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(manifest_path, "w") as out:
        for (dst, text), duration in zip(rows, tqdm(pool.map(convert_clip, jobs), total=len(jobs), desc=split)):
            if not text:
                continue
            out.write(json.dumps({"audio_filepath": str(dst), "text": text, "duration": duration}, ensure_ascii=False) + "\n")
            written += 1
    return written


def prepare_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending_splits = [
        split for split in args.splits
        if not (output_dir / f"{split}_manifest.json").exists()
    ]

    corpus_dir = None
    if pending_splits:
        corpus_dir = find_existing_corpus(output_dir)
        if corpus_dir:
            print(f"Found existing extracted corpus at {corpus_dir}, skipping download/extraction")
        elif args.skip_download:
            corpus_dir = next(p for p in output_dir.iterdir() if p.is_dir())
        else:
            archive_path = download_dataset(args.dataset_id, args.api_key, output_dir)
            corpus_dir = extract_dataset(archive_path, output_dir)

    counts = {}
    for split in args.splits:
        manifest_path = output_dir / f"{split}_manifest.json"
        if split not in pending_splits:
            with open(manifest_path) as f:
                counts[split] = sum(1 for _ in f)
            print(f"{split}: manifest already exists -> {manifest_path} ({counts[split]} utterances), skipping")
            continue
        counts[split] = build_manifest(corpus_dir, split, output_dir / "wavs", args.workers)
        print(f"{split}: wrote {counts[split]} utterances -> {manifest_path}")
    return counts
