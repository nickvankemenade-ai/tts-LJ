"""Audio constants, log-mel features and dependency-free WAV I/O."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 16_000
N_FFT = 1024
HOP = 256
N_MELS = 80


class LogMel(torch.nn.Module):
    """Log-mel spectrogram with exactly ``len(wav) // HOP`` frames.

    Always computed in float32, also under autocast.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT,
                 hop: int = HOP, n_mels: int = N_MELS):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft, hop_length=hop,
            f_min=0.0, f_max=sample_rate / 2, n_mels=n_mels, power=1.0,
            center=False, norm="slaney", mel_scale="slaney",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, T) or (B, 1, T) waveform -> (B, n_mels, T // hop)."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            pad = (self.n_fft - self.hop) // 2
            wav = F.pad(wav.float().unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
            return torch.log(torch.clamp(self.mel(wav), min=1e-5))


def read_wav(path_or_file) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file (mono or multi-channel, mixed down) as float32."""
    with wave.open(path_or_file if not isinstance(path_or_file, Path) else str(path_or_file), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def write_wav(path, audio, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a float waveform in [-1, 1] as 16-bit PCM mono WAV."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    audio = np.clip(np.asarray(audio).reshape(-1), -1.0, 1.0)
    if hasattr(path, "write"):  # file-like object, e.g. io.BytesIO
        target = path
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        target = str(path)
    with wave.open(target, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((audio * 32767.0).astype("<i2").tobytes())


FLOOR = math.log(1e-5)  # log-mel value of silence


def bin_frequencies(n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT) -> list[float]:
    """Approximate centre frequency (Hz) of each mel bin."""
    fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, 0.0, sample_rate / 2, n_mels, sample_rate,
                                               norm="slaney", mel_scale="slaney")
    return (fb.argmax(dim=0).float() * sample_rate / n_fft).tolist()


def to_model_audio(audio: np.ndarray, sr: int) -> torch.Tensor:
    """Any-rate float audio -> 16 kHz, cut to a multiple of HOP, peak-limited to 0.9."""
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav[: len(wav) - len(wav) % HOP]
    peak = wav.abs().max()
    if peak > 0.9:
        wav = wav * (0.9 / peak)
    return wav
