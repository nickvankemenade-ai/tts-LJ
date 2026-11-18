"""HiFi-GAN-style vocoder (Kong et al., 2020).

Generator presets (all upsample by 256 = HOP):
  v1  ~14M params, 512 channels, ResBlock1   (best quality, slowest)
  v2  ~0.9M params, 128 channels, ResBlock1
  v3  ~1.5M params, 256 channels, ResBlock2  (fast, lowest quality ceiling)

Discriminator presets:
  hifigan  multi-period (5 x full width) + multi-scale (3), ~70M params, as in the paper
  light    multi-period at half width + multi-resolution spectrogram discriminators
           (UnivNet / Vocos / BigVGAN style), a fraction of the compute
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU = 0.1


def get_padding(k: int, d: int = 1) -> int:
    return (k * d - d) // 2


# --------------------------------------------------------------------------- generator

@dataclass
class GeneratorConfig:
    n_mels: int = 80
    upsample_initial_channel: int = 256
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 4])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 8])
    resblock: int = 2  # 1 = ResBlock1 (two convs per dilation), 2 = ResBlock2 (one)
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    resblock_dilation_sizes: list[list[int]] = field(default_factory=lambda: [[1, 2], [2, 6], [3, 12]])

    def to_dict(self) -> dict:
        return asdict(self)


GENERATOR_PRESETS = {
    "v1": dict(upsample_initial_channel=512, upsample_rates=[8, 8, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4],
               resblock=1, resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5]] * 3),
    "v2": dict(upsample_initial_channel=128, upsample_rates=[8, 8, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4],
               resblock=1, resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5]] * 3),
    "v3": dict(),  # the dataclass defaults
}


class ResBlock1(nn.Module):
    """Two convolutions (dilated, then plain) per residual step."""

    def __init__(self, ch: int, k: int, dilations: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(nn.Conv1d(ch, ch, k, dilation=d, padding=get_padding(k, d))) for d in dilations
        )
        self.convs2 = nn.ModuleList(
            weight_norm(nn.Conv1d(ch, ch, k, padding=get_padding(k, 1))) for _ in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
        return x


class ResBlock2(nn.Module):
    """One dilated convolution per residual step."""

    def __init__(self, ch: int, k: int, dilations: list[int]):
        super().__init__()
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv1d(ch, ch, k, dilation=d, padding=get_padding(k, d))) for d in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(F.leaky_relu(x, LRELU))
        return x


ResBlock = ResBlock2  # backwards-compatible name


class Generator(nn.Module):
    def __init__(self, cfg: GeneratorConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or GeneratorConfig()
        block = ResBlock1 if cfg.resblock == 1 else ResBlock2
        ch = cfg.upsample_initial_channel
        self.conv_pre = weight_norm(nn.Conv1d(cfg.n_mels, ch, 7, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            c_in, c_out = ch // 2 ** i, ch // 2 ** (i + 1)
            self.ups.append(weight_norm(nn.ConvTranspose1d(c_in, c_out, k, u, padding=(k - u) // 2)))
            for rk, rd in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(block(c_out, rk, rd))
        self.conv_post = weight_norm(nn.Conv1d(c_out, 1, 7, padding=3))
        self.num_kernels = len(cfg.resblock_kernel_sizes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, frames) -> (B, 1, frames * prod(upsample_rates))."""
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU))
            xs = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


# ----------------------------------------------------------------------- discriminators

class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, width: float = 1.0):
        super().__init__()
        self.period = period
        chans = [1] + [max(8, int(c * width)) for c in (32, 128, 512, 1024)]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chans[i], chans[i + 1], (5, 1), (3, 1), padding=(2, 0))) for i in range(4)
        )
        self.convs.append(weight_norm(nn.Conv2d(chans[-1], chans[-1], (5, 1), 1, padding=(2, 0))))
        self.conv_post = weight_norm(nn.Conv2d(chans[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        specs = [(1, 128, 15, 1, 1, 7), (128, 128, 41, 2, 4, 20), (128, 256, 41, 2, 16, 20),
                 (256, 512, 41, 4, 16, 20), (512, 1024, 41, 4, 16, 20), (1024, 1024, 41, 1, 16, 20),
                 (1024, 1024, 5, 1, 1, 2)]
        self.convs = nn.ModuleList(weight_norm(nn.Conv1d(i, o, k, s, groups=g, padding=p))
                                   for i, o, k, s, g, p in specs)
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class ResolutionDiscriminator(nn.Module):
    """2-D convs over a linear-magnitude STFT at one resolution (UnivNet's MRD)."""

    def __init__(self, n_fft: int, hop: int, win: int, ch: int = 32):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, ch, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.stft(x.squeeze(1).float(), self.n_fft, self.hop, window=self.window,
                              win_length=self.window.numel(), center=True, return_complex=True).abs()
        x = spec.transpose(1, 2).unsqueeze(1)  # (B, 1, frames, freq)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiDiscriminator(nn.Module):
    """``kind="hifigan"``: MPD + MSD as in the paper. ``kind="light"``: half-width MPD + MRD."""

    def __init__(self, kind: str = "hifigan"):
        super().__init__()
        self.kind = kind
        periods = (2, 3, 5, 7, 11)
        if kind == "hifigan":
            self.period_discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])
            # The paper puts spectral norm on the first scale discriminator; PyTorch's spectral_norm
            # parametrization fails under bf16 autocast (torch 2.3), so all three use weight norm.
            self.other_discs = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])
            self.pool = nn.AvgPool1d(4, 2, padding=2)
        elif kind == "light":
            self.period_discs = nn.ModuleList([PeriodDiscriminator(p, width=0.5) for p in periods])
            self.other_discs = nn.ModuleList([ResolutionDiscriminator(n, h, w) for n, h, w in
                                              ((512, 64, 256), (1024, 128, 512), (2048, 256, 1024))])
        else:
            raise ValueError(kind)

    def forward(self, real, fake):
        """Returns (outputs_real, outputs_fake, feature_maps_real, feature_maps_fake)."""
        outs_r, outs_f, fmaps_r, fmaps_f = [], [], [], []

        def run(d, xr, xf):
            o_r, f_r = d(xr)
            o_f, f_f = d(xf)
            outs_r.append(o_r)
            outs_f.append(o_f)
            fmaps_r.append(f_r)
            fmaps_f.append(f_f)

        for d in self.period_discs:
            run(d, real, fake)
        xr, xf = real, fake
        for i, d in enumerate(self.other_discs):
            if self.kind == "hifigan" and i > 0:  # scales x2 and x4 see average-pooled audio
                xr, xf = self.pool(xr), self.pool(xf)
            run(d, xr, xf)
        return outs_r, outs_f, fmaps_r, fmaps_f


# ------------------------------------------------------------------------------ losses

def discriminator_loss(outs_r, outs_f) -> torch.Tensor:
    return sum(torch.mean((1 - r.float()) ** 2) + torch.mean(f.float() ** 2) for r, f in zip(outs_r, outs_f))


def generator_adv_loss(outs_f) -> torch.Tensor:
    return sum(torch.mean((1 - f.float()) ** 2) for f in outs_f)


def feature_matching_loss(fmaps_r, fmaps_f) -> torch.Tensor:
    return 2 * sum(torch.mean(torch.abs(r.float() - f.float()))
                   for fr, ff in zip(fmaps_r, fmaps_f) for r, f in zip(fr, ff))
