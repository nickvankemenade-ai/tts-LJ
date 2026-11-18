"""Checkpoint loading and small helpers shared by the scripts."""

from __future__ import annotations

from pathlib import Path

import torch

from .audio import LogMel
from .gan import Generator, GeneratorConfig

# data/ and runs/ live in the repo root (both gitignored)
DEFAULT_WORK_DIR = Path(__file__).resolve().parents[1]


def load_vocoder(path, device="cpu") -> tuple[Generator, dict]:
    """GAN vocoder checkpoint -> (generator in eval mode, raw checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Generator(GeneratorConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def vocode(generator: Generator, mel: torch.Tensor) -> torch.Tensor:
    """mel (B, n_mels, F) -> waveform (B, F * HOP)."""
    return generator(mel).squeeze(1)


@torch.no_grad()
def mel_distance(melfn: LogMel, pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean absolute log-mel difference (lower is better)."""
    n = min(pred.shape[-1], ref.shape[-1])
    return torch.mean(torch.abs(melfn(pred[..., :n]) - melfn(ref[..., :n]))).item()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
