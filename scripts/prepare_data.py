"""Build the 16 kHz training set, keeping disk use small.

Sources (choose one):
  --download            stream LJSpeech-1.1 (~2.6 GB .tar.bz2) straight from the web and
                        decompress it on the fly. The archive is never written to disk.
  --archive PATH        an LJSpeech-style .tar.bz2 / .tar.gz you already have.
  --wav-dir PATH        a folder of 16-bit PCM .wav files, e.g. CSS10 Dutch.

Output in --data-dir: train.bin, train_index.npy, val.bin, val_index.npy, meta.json, and for
LJSpeech-style archives transcripts.json (clip names and normalized text, in index order).
Disk: about 115 MB per hour of audio (full LJSpeech, ~24 h, is about 2.7 GB).
Use --max-hours to cap the training set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tms.audio import SAMPLE_RATE, read_wav  # noqa: E402

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"  # repo root, gitignored


def iter_archive(fileobj, mode: str, metadata: dict):
    """Yields (name, wav file); fills ``metadata`` (name -> normalized text) from metadata.csv."""
    with tarfile.open(fileobj=fileobj, mode=mode) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith("metadata.csv"):
                text = tar.extractfile(member).read().decode("utf-8")
                rows = csv.reader(io.StringIO(text), delimiter="|", quoting=csv.QUOTE_NONE)
                metadata.update({r[0]: r[2] for r in rows if len(r) >= 3})
            elif member.isfile() and member.name.endswith(".wav"):
                yield Path(member.name).stem, io.BytesIO(tar.extractfile(member).read())


def iter_wav_dir(wav_dir: Path):
    for path in sorted(wav_dir.rglob("*.wav")):
        yield path.stem, path


def is_val(name: str, val_percent: float) -> bool:
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 10_000
    return h < val_percent * 100


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--download", action="store_true", help="stream LJSpeech from the web")
    src.add_argument("--archive", type=Path)
    src.add_argument("--wav-dir", type=Path)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--max-hours", type=float, default=None, help="cap on training audio")
    p.add_argument("--val-percent", type=float, default=1.0)
    p.add_argument("--max-val", type=int, default=100)
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.data_dir).free / 1e9
    print(f"output: {args.data_dir}  (free disk: {free_gb:.1f} GB)")

    metadata: dict[str, str] = {}
    if args.download:
        print(f"streaming {LJSPEECH_URL} (nothing but the 16 kHz output is written to disk)")
        response = urllib.request.urlopen(LJSPEECH_URL)
        items = iter_archive(response, "r|bz2", metadata)
    elif args.archive:
        mode = "r|bz2" if args.archive.suffix == ".bz2" else "r|gz"
        items = iter_archive(open(args.archive, "rb"), mode, metadata)
    else:
        items = iter_wav_dir(args.wav_dir)

    files = {s: open(args.data_dir / f"{s}.bin", "wb") for s in ("train", "val")}
    index = {"train": [], "val": []}
    names = {"train": [], "val": []}
    pos = {"train": 0, "val": 0}
    resamplers: dict[int, torchaudio.transforms.Resample] = {}
    max_train = None if args.max_hours is None else int(args.max_hours * 3600 * SAMPLE_RATE)
    t0, n = time.time(), 0
    try:
        for name, src_file in items:
            wav, sr = read_wav(src_file)
            if sr != SAMPLE_RATE:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = resamplers[sr](torch.from_numpy(wav)).numpy()
            split = "val" if is_val(name, args.val_percent) and len(index["val"]) < args.max_val else "train"
            if split == "train" and max_train is not None and pos["train"] >= max_train:
                if len(index["val"]) >= args.max_val or args.wav_dir is None:
                    break
                continue
            pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
            files[split].write(pcm.tobytes())
            index[split].append((pos[split], len(pcm)))
            names[split].append(name)
            pos[split] += len(pcm)
            n += 1
            if n % 500 == 0:
                hrs = pos["train"] / SAMPLE_RATE / 3600
                print(f"  {n} files, train {hrs:.2f} h, val {len(index['val'])} utts, {time.time() - t0:.0f}s")
    finally:
        for f in files.values():
            f.close()

    for s in ("train", "val"):
        np.save(args.data_dir / f"{s}_index.npy", np.asarray(index[s], dtype=np.int64).reshape(-1, 2))
    meta = {
        "sample_rate": SAMPLE_RATE,
        "train_hours": pos["train"] / SAMPLE_RATE / 3600,
        "train_utterances": len(index["train"]),
        "val_utterances": len(index["val"]),
        "source": LJSPEECH_URL if args.download else str(args.archive or args.wav_dir),
    }
    (args.data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if metadata:
        transcripts = {s: [[n, metadata.get(n, "")] for n in names[s]] for s in names}
        (args.data_dir / "transcripts.json").write_text(json.dumps(transcripts, indent=0, ensure_ascii=False),
                                                        encoding="utf-8")
        missing = sum(1 for s in names for n in names[s] if n not in metadata)
        print(f"transcripts.json written ({missing} clips without text)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
