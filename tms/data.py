"""Compact on-disk dataset: one int16 ``<split>.bin`` file plus ``<split>_index.npy``.

The index holds one (start, length) row per utterance, in samples. Audio is
stored once at 16 kHz and mel features are computed on the GPU during
training, so the dataset takes about 115 MB per hour of audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .audio import HOP


def open_split(data_dir, split: str) -> tuple[np.memmap, np.ndarray]:
    data_dir = Path(data_dir)
    audio = np.memmap(data_dir / f"{split}.bin", dtype="<i2", mode="r")
    index = np.load(data_dir / f"{split}_index.npy")
    return audio, index


class SegmentDataset(torch.utils.data.Dataset):
    """Random fixed-length segments from random utterances.

    The memmap is opened lazily in each worker, so nothing large gets pickled
    when DataLoader workers start (Windows uses spawn).
    """

    def __init__(self, data_dir, split: str = "train", segment: int = 8192,
                 virtual_len: int = 10_000_000):
        assert segment % HOP == 0, "segment must be a multiple of HOP"
        self.data_dir, self.split, self.segment = str(data_dir), split, segment
        self.virtual_len = virtual_len
        self._audio = None
        index = np.load(Path(data_dir) / f"{split}_index.npy")
        self.index = index[index[:, 1] >= segment]
        if len(self.index) == 0:
            raise ValueError(f"no utterances of at least {segment} samples in {split}")

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, i: int) -> torch.Tensor:
        if self._audio is None:
            self._audio, _ = open_split(self.data_dir, self.split)
        rng = np.random.default_rng(i * 7919 + torch.initial_seed() % 100_003)
        start, length = self.index[rng.integers(len(self.index))]
        offset = start + rng.integers(0, length - self.segment + 1)
        seg = np.asarray(self._audio[offset:offset + self.segment], dtype=np.float32) / 32768.0
        return torch.from_numpy(seg)


class TextAudioDataset(torch.utils.data.Dataset):
    """Whole utterances with their transcripts (``transcripts.json`` from prepare_data.py).

    Items are (token ids, float waveform cut to a multiple of HOP).
    """

    def __init__(self, data_dir, split: str = "train", max_seconds: float = 12.0, sample_rate: int = 16_000):
        import json

        from .tts import text_to_ids

        self.data_dir, self.split = str(data_dir), split
        self._audio = None
        index = np.load(Path(data_dir) / f"{split}_index.npy")
        texts = json.loads((Path(data_dir) / "transcripts.json").read_text(encoding="utf-8"))[split]
        assert len(texts) == len(index), "transcripts.json does not match the index; rerun prepare_data.py"
        self.items = []
        for (start, length), (_, text) in zip(index, texts):
            ids = text_to_ids(text)
            frames = int(length) // HOP
            if frames >= len(ids) and length <= max_seconds * sample_rate:
                self.items.append((int(start), frames * HOP, ids, text))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        if self._audio is None:
            self._audio, _ = open_split(self.data_dir, self.split)
        start, length, ids, _ = self.items[i]
        wav = np.asarray(self._audio[start:start + length], dtype=np.float32) / 32768.0
        return torch.tensor(ids, dtype=torch.long), torch.from_numpy(wav)

    def length_batches(self, batch_size: int, rng: np.random.Generator) -> list[list[int]]:
        """Batches of similar length (little padding), in random order."""
        jitter = rng.integers(0, 16_000, size=len(self.items))  # up to 1 s, so batches differ per epoch
        order = np.argsort(np.array([it[1] for it in self.items]) + jitter)
        batches = [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]
        rng.shuffle(batches)
        return batches


def collate_text_audio(batch):
    """-> ids (B, Tx), id lengths, padded waveforms (B, T), mel frame lengths."""
    ids, wavs = zip(*batch)
    x_len = torch.tensor([len(x) for x in ids])
    w_len = torch.tensor([len(w) for w in wavs])
    x = torch.zeros(len(ids), int(x_len.max()), dtype=torch.long)
    w = torch.zeros(len(wavs), int(w_len.max()))
    for i, (a, b) in enumerate(zip(ids, wavs)):
        x[i, :len(a)] = a
        w[i, :len(b)] = b
    return x, x_len, w, w_len // HOP


def load_utterances(data_dir, split: str = "val", n: int = 8,
                    max_seconds: float = 4.0, sample_rate: int = 16_000) -> list[torch.Tensor]:
    """First ``n`` utterances of a split, cut to ``max_seconds`` and a multiple of HOP."""
    audio, index = open_split(data_dir, split)
    out = []
    for start, length in index[:n]:
        length = min(int(length), int(max_seconds * sample_rate))
        length -= length % HOP
        out.append(torch.from_numpy(np.asarray(audio[start:start + length], dtype=np.float32) / 32768.0))
    return out
