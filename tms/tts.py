"""Matcha-TTS-style text-to-mel model (Mehta et al., 2024), trained from scratch on characters.

  text ids -> transformer encoder -> per-token mel mean ``mu`` and log-duration
  alignment     monotonic alignment search (MAS) between ``mu`` and the target mel, every step;
                no external aligner. The duration predictor learns the MAS durations.
  decoder       a conditional flow-matching network turns noise into the mel, given ``mu``
                upsampled to frame rate. A few Euler steps at inference.

Output is the same 80-band log-mel as ``tms.audio.LogMel``, so the GAN vocoder
renders it directly.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import numba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------ text

PAD, BLANK = 0, 1
CHARS = " !'(),-.:;?\"abcdefghijklmnopqrstuvwxyz"
SYMBOL_ID = {c: i + 2 for i, c in enumerate(CHARS)}
N_VOCAB = len(CHARS) + 2
_REPLACE = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "[": "(", "]": ")",
            "ü": "u", "é": "e", "è": "e", "ê": "e", "à": "a", "â": "a"}


def clean_text(text: str) -> str:
    text = text.lower()
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    text = "".join(c for c in text if c in SYMBOL_ID)
    return re.sub(r"\s+", " ", text).strip()


def text_to_ids(text: str) -> list[int]:
    """Characters with a blank token between every pair (and at both ends), as in Glow-TTS/Matcha."""
    ids = [BLANK]
    for c in clean_text(text):
        ids += [SYMBOL_ID[c], BLANK]
    return ids


# ------------------------------------------------------------------------------ config

@dataclass
class TTSConfig:
    n_vocab: int = N_VOCAB
    n_mels: int = 80
    enc_channels: int = 192
    enc_ffn: int = 768
    enc_layers: int = 6
    enc_heads: int = 2
    enc_kernel: int = 3
    dropout: float = 0.1
    dp_channels: int = 256
    dec_channels: int = 256
    dec_groups: int = 4          # each group: 2 residual conv blocks + 1 transformer block
    dec_heads: int = 4
    sigma_min: float = 1e-4
    segment_frames: int = 128    # decoder is trained on random crops of this many frames (~2 s)
    mel_mean: float = 0.0        # global log-mel statistics, set from the data before training
    mel_std: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


TTS_PRESETS = {
    "base": {},
    "small": dict(enc_channels=160, enc_ffn=512, enc_layers=4, dp_channels=192, dec_channels=192, dec_groups=3),
}


# ------------------------------------------------------------------------------ blocks

def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = max_len or int(lengths.max())
    return torch.arange(max_len, device=lengths.device)[None] < lengths[:, None]


class ChannelNorm(nn.Module):
    """LayerNorm over channels of a (B, C, T) tensor."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on (B, H, T, D)."""
    d, t = x.shape[-1], x.shape[-2]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, ch: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(ch, 3 * ch)
        self.out = nn.Linear(ch, ch)

    def forward(self, x, mask):
        """x (B, C, T), mask (B, T) bool."""
        b, c, t = x.shape
        q, k, v = self.qkv(x.transpose(1, 2)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, c)).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Post-norm block with a convolutional feed-forward (as in Glow-TTS / Matcha encoders)."""

    def __init__(self, ch: int, ffn: int, heads: int, kernel: int, dropout: float):
        super().__init__()
        self.attn = Attention(ch, heads, dropout)
        self.norm1, self.norm2 = ChannelNorm(ch), ChannelNorm(ch)
        self.ff1 = nn.Conv1d(ch, ffn, kernel, padding=kernel // 2)
        self.ff2 = nn.Conv1d(ffn, ch, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        m = mask[:, None].to(x.dtype)
        x = self.norm1(x + self.drop(self.attn(x, mask)))
        y = self.ff2(self.drop(F.relu(self.ff1(x * m))) * m) * m
        return self.norm2(x + self.drop(y)) * m


class TextEncoder(nn.Module):
    def __init__(self, cfg: TTSConfig):
        super().__init__()
        ch = cfg.enc_channels
        self.emb = nn.Embedding(cfg.n_vocab, ch)
        nn.init.normal_(self.emb.weight, 0.0, ch ** -0.5)
        self.prenet = nn.ModuleList(nn.Conv1d(ch, ch, 5, padding=2) for _ in range(3))
        self.prenet_norms = nn.ModuleList(ChannelNorm(ch) for _ in range(3))
        self.blocks = nn.ModuleList(TransformerBlock(ch, cfg.enc_ffn, cfg.enc_heads, cfg.enc_kernel, cfg.dropout)
                                    for _ in range(cfg.enc_layers))
        self.proj_mu = nn.Conv1d(ch, cfg.n_mels, 1)
        dp = cfg.dp_channels
        self.dp = nn.ModuleList([nn.Conv1d(ch, dp, 3, padding=1), nn.Conv1d(dp, dp, 3, padding=1)])
        self.dp_norms = nn.ModuleList([ChannelNorm(dp), ChannelNorm(dp)])
        self.dp_out = nn.Conv1d(dp, 1, 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = math.sqrt(ch)

    def forward(self, ids, lengths):
        """-> mu (B, n_mels, T), log-duration (B, T), mask (B, T)."""
        mask = sequence_mask(lengths, ids.shape[1])
        m = mask[:, None].float()
        x = self.emb(ids).transpose(1, 2) * self.scale * m
        for conv, norm in zip(self.prenet, self.prenet_norms):
            x = x + self.drop(F.relu(norm(conv(x * m))))
        x = x * m
        for block in self.blocks:
            x = block(x, mask)
        mu = self.proj_mu(x) * m
        h = x.detach()  # the duration predictor does not shape the encoder
        for conv, norm in zip(self.dp, self.dp_norms):
            h = self.drop(norm(F.relu(conv(h * m))))
        logw = self.dp_out(h * m).squeeze(1) * mask
        return mu, logw, mask


class TimeEmbedding(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.ch = ch
        self.mlp = nn.Sequential(nn.Linear(ch, 4 * ch), nn.SiLU(), nn.Linear(4 * ch, ch))

    def forward(self, t):
        half = self.ch // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = 1000 * t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(8, ch), nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.film = nn.Linear(ch, 2 * ch)

    def forward(self, x, m, temb):
        scale, shift = self.film(F.silu(temb))[:, :, None].chunk(2, dim=1)
        h = self.conv1(F.mish(self.norm1(x)) * m)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.mish(h) * m)
        return (x + h) * m


class FlowDecoder(nn.Module):
    """Velocity network v(x_t, t | mu): residual conv blocks + transformer blocks at frame rate."""

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        ch = cfg.dec_channels
        self.inp = nn.Conv1d(2 * cfg.n_mels, ch, 3, padding=1)
        self.temb = TimeEmbedding(ch)
        self.groups = nn.ModuleList(
            nn.ModuleList([ResBlock(ch), ResBlock(ch), TransformerBlock(ch, 4 * ch, cfg.dec_heads, 1, 0.0)])
            for _ in range(cfg.dec_groups))
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.Mish(), nn.Conv1d(ch, cfg.n_mels, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, mask, mu, t):
        m = mask[:, None].to(x.dtype)
        temb = self.temb(t)
        h = self.inp(torch.cat([x, mu], dim=1)) * m
        for r1, r2, attn in self.groups:
            h = r2(r1(h, m, temb), m, temb)
            h = attn(h, mask)
        return self.out(h) * m


# ------------------------------------------------------------------------------ alignment

@numba.njit(parallel=True, cache=True)
def _mas_batch(values, paths, t_xs, t_ys):
    for b in numba.prange(values.shape[0]):
        v, t_x, t_y = values[b], t_xs[b], t_ys[b]
        for j in range(t_y):  # forward pass, in place: v[i, j] becomes the best score ending at (i, j)
            for i in range(max(0, t_x + j - t_y), min(t_x, j + 1)):
                if j == 0:
                    best = 0.0 if i == 0 else -1e9
                else:
                    stay = v[i, j - 1] if i < j else -1e9
                    move = v[i - 1, j - 1] if i > 0 else -1e9
                    best = max(stay, move)
                v[i, j] += best
        i = t_x - 1
        for j in range(t_y - 1, -1, -1):  # backtrack
            paths[b, i, j] = 1
            if i > 0 and (i == j or v[i, j - 1] < v[i - 1, j - 1]):
                i -= 1


@torch.no_grad()
def monotonic_alignment(logp: torch.Tensor, x_len: torch.Tensor, y_len: torch.Tensor) -> torch.Tensor:
    """Most likely monotonic hard alignment. logp (B, Tx, Ty) -> 0/1 path of the same shape."""
    values = logp.float().cpu().numpy().copy()
    paths = np.zeros(values.shape, dtype=np.float32)
    _mas_batch(values, paths, x_len.cpu().numpy().astype(np.int64), y_len.cpu().numpy().astype(np.int64))
    return torch.from_numpy(paths).to(logp.device)


def durations_to_path(durations: torch.Tensor, x_mask: torch.Tensor, y_len_max: int) -> torch.Tensor:
    """Integer durations (B, Tx) -> 0/1 path (B, Tx, Ty)."""
    cum = torch.cumsum(durations * x_mask, dim=1)
    frames = torch.arange(y_len_max, device=durations.device)[None, None]
    return ((frames < cum[..., None]) & (frames >= (cum - durations)[..., None]) & x_mask[..., None]).float()


# ------------------------------------------------------------------------------ model

class MatchaTTS(nn.Module):
    def __init__(self, cfg: TTSConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or TTSConfig()
        self.encoder = TextEncoder(cfg)
        self.decoder = FlowDecoder(cfg)

    def normalize(self, mel):
        return (mel - self.cfg.mel_mean) / self.cfg.mel_std

    def denormalize(self, mel):
        return mel * self.cfg.mel_std + self.cfg.mel_mean

    def loss(self, ids, x_len, mel, y_len) -> dict[str, torch.Tensor]:
        """ids (B, Tx), mel (B, n_mels, Ty) un-normalized log-mel. Returns the three loss terms."""
        cfg = self.cfg
        mu_x, logw, x_mask = self.encoder(ids, x_len)
        y = self.normalize(mel[..., :int(y_len.max())])
        y_mask = sequence_mask(y_len, y.shape[-1])
        y = y * y_mask[:, None]

        attn = monotonic_alignment(_log_likelihood(mu_x, y), x_len, y_len)

        logw_target = torch.log(1e-8 + attn.sum(-1)) * x_mask
        dur_loss = ((logw.float() - logw_target) ** 2).sum() / x_len.sum()

        mu_y = torch.bmm(mu_x.float(), attn)  # (B, n_mels, Ty)
        prior_loss = ((0.5 * ((y.float() - mu_y) ** 2 + math.log(2 * math.pi))) * y_mask[:, None]).sum() \
            / (y_mask.sum() * cfg.n_mels)

        # flow matching on a random crop per utterance
        seg = min(cfg.segment_frames or y.shape[-1], y.shape[-1])
        starts = (torch.rand(len(y_len), device=y.device) * (y_len - seg).clamp(min=0).float()).long()
        idx = starts[:, None] + torch.arange(seg, device=y.device)[None]
        gather = lambda a: torch.gather(a, 2, idx[:, None].expand(-1, a.shape[1], -1))  # noqa: E731
        y_s, mu_s = gather(y.float()), gather(mu_y)
        mask_s = torch.gather(y_mask, 1, idx)
        t = torch.rand(len(y_len), device=y.device)
        z = torch.randn_like(y_s)
        tt = t[:, None, None]
        x_t = (1 - (1 - cfg.sigma_min) * tt) * z + tt * y_s
        target = y_s - (1 - cfg.sigma_min) * z
        pred = self.decoder(x_t, mask_s, mu_s, t).float()
        fm_loss = (((pred - target) ** 2) * mask_s[:, None]).sum() / (mask_s.sum() * cfg.n_mels)
        return {"prior": prior_loss, "dur": dur_loss, "fm": fm_loss}

    @torch.no_grad()
    def synthesize(self, ids, x_len, steps: int = 20, temperature: float = 0.3,
                   length_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """-> log-mel (B, n_mels, Ty) and frame lengths (B,)."""
        mu_x, logw, x_mask = self.encoder(ids, x_len)
        w = torch.ceil(torch.exp(logw.float()) * x_mask * length_scale)
        y_len = w.sum(1).clamp(min=1).long()
        attn = durations_to_path(w, x_mask, int(y_len.max()))
        mu_y = torch.bmm(mu_x.float(), attn)
        y_mask = sequence_mask(y_len, mu_y.shape[-1])
        x = torch.randn_like(mu_y) * temperature
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((len(x),), k * dt, device=x.device)
            x = x + dt * self.decoder(x, y_mask, mu_y, t).float()
        return self.denormalize(x) * y_mask[:, None] + math.log(1e-5) * ~y_mask[:, None], y_len


def _log_likelihood(mu_x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log N(y_j; mu_i, I) up to a constant, for every token i and frame j: (B, Tx, Ty)."""
    with torch.no_grad():
        mu_f, y_f = mu_x.float(), y.float()
        return (-0.5 * (mu_f ** 2).sum(1)[:, :, None] + torch.einsum("bci,bcj->bij", mu_f, y_f)
                - 0.5 * (y_f ** 2).sum(1)[:, None, :])


def load_tts(path, device="cpu") -> tuple[MatchaTTS, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MatchaTTS(TTSConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck
