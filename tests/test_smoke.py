import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tms.audio import HOP, LogMel, read_wav, to_model_audio, write_wav  # noqa: E402
from tms.common import load_vocoder, vocode  # noqa: E402
from tms.data import SegmentDataset, load_utterances  # noqa: E402
from tms.gan import GENERATOR_PRESETS, Generator, GeneratorConfig, MultiDiscriminator  # noqa: E402


def test_mel_frames():
    mel = LogMel()(torch.randn(2, 40 * HOP))
    assert mel.shape == (2, 80, 40)


def test_to_model_audio():
    wav = to_model_audio(np.sin(np.arange(44100) / 10).astype(np.float32) * 2, 44100)
    assert len(wav) % HOP == 0 and abs(len(wav) - 16000) <= HOP and wav.abs().max() <= 0.9 + 1e-6


def test_generator_and_discriminator_shapes():
    for preset in GENERATOR_PRESETS.values():
        cfg = GeneratorConfig(**{**preset, "upsample_initial_channel": 32})
        assert Generator(cfg)(torch.randn(2, 80, 10)).shape == (2, 1, 10 * HOP)
    wav = Generator()(torch.randn(2, 80, 10))
    assert wav.shape == (2, 1, 10 * HOP)
    for kind in ("hifigan", "light"):
        outs_r, outs_f, fr, ff = MultiDiscriminator(kind)(wav.detach(), wav.detach())
        assert len(outs_r) == len(outs_f) == len(fr) == len(ff) == 8


def _make_wavs(folder: Path, n: int = 12):
    folder.mkdir(parents=True)
    for i in range(n):
        t = np.arange(22050) / 22050
        write_wav(folder / f"utt{i:03d}.wav", 0.3 * np.sin(2 * np.pi * (150 + 10 * i) * t), 22050)


def test_prepare_and_train_gan(tmp_path):
    wav_dir, data_dir = tmp_path / "wavs", tmp_path / "data"
    _make_wavs(wav_dir)
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], check=True, cwd=ROOT)  # noqa: E731
    run("scripts/prepare_data.py", "--wav-dir", wav_dir, "--data-dir", data_dir, "--val-percent", "30")
    audio, sr = read_wav(wav_dir / "utt000.wav")
    assert sr == 22050 and len(audio) == 22050
    ds = SegmentDataset(data_dir, "train", segment=4096)
    assert ds[0].shape == (4096,)
    assert len(load_utterances(data_dir, "val", n=2)) >= 1

    out = tmp_path / "gan"
    common = ["scripts/train_gan.py", "--small", "--data-dir", data_dir, "--out", out, "--batch-size", "2",
              "--segment", "4096", "--workers", "0", "--mel-only-minutes", "0", "--log-every", "1", "--no-amp"]
    run(*common, "--max-steps", "3")
    gen, ck = load_vocoder(out / "latest.pt")
    assert ck["kind"] == "gan" and vocode(gen, torch.randn(1, 80, 4)).shape == (1, 4 * HOP)
    assert (out / "resume.pt").exists() and (out / "best.pt").exists()
    run(*common, "--max-steps", "5", "--resume")  # resuming continues from the saved step
    assert torch.load(out / "latest.pt", weights_only=False)["step"] == 5
