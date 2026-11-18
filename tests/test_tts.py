import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tms.audio import HOP  # noqa: E402
from tms.tts import (BLANK, MatchaTTS, TTSConfig, clean_text, durations_to_path,  # noqa: E402
                         monotonic_alignment, text_to_ids)


def test_text():
    assert clean_text("Hello,  “World” — 2!") == 'hello, "world" - !'
    ids = text_to_ids("ab")
    assert ids[0] == ids[2] == ids[4] == BLANK and len(ids) == 5


def _brute_force(logp, t_x, t_y):
    """Best monotonic path by exhaustive search over durations (tiny sizes only)."""
    import itertools
    best, best_path = -np.inf, None
    for cuts in itertools.combinations(range(1, t_y), t_x - 1):
        bounds = (0, *cuts, t_y)
        score = sum(logp[i, bounds[i]:bounds[i + 1]].sum() for i in range(t_x))
        if score > best:
            best, best_path = score, bounds
    path = np.zeros((t_x, t_y), dtype=np.float32)
    for i in range(t_x):
        path[i, best_path[i]:best_path[i + 1]] = 1
    return path


def test_mas_matches_brute_force():
    rng = np.random.default_rng(0)
    for t_x, t_y in [(1, 4), (3, 3), (3, 7), (4, 9)]:
        logp = rng.normal(size=(t_x, t_y)).astype(np.float32)
        padded = torch.full((1, 5, 10), -1e4)
        padded[0, :t_x, :t_y] = torch.from_numpy(logp)
        path = monotonic_alignment(padded, torch.tensor([t_x]), torch.tensor([t_y]))[0, :t_x, :t_y].numpy()
        assert (path == _brute_force(logp, t_x, t_y)).all()
        assert path.sum() == t_y


def test_durations_to_path():
    path = durations_to_path(torch.tensor([[2.0, 1.0, 0.0]]), torch.tensor([[True, True, False]]), 4)
    assert path[0].tolist() == [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]


def test_model_loss_and_synthesis():
    cfg = TTSConfig(enc_channels=32, enc_ffn=64, enc_layers=1, dp_channels=32, dec_channels=32, dec_groups=1,
                    segment_frames=8)
    m = MatchaTTS(cfg)
    ids = torch.tensor([text_to_ids("hello there"), text_to_ids("hi") + [0] * 18])
    x_len = torch.tensor([len(text_to_ids("hello there")), len(text_to_ids("hi"))])
    losses = m.loss(ids, x_len, torch.randn(2, 80, 40), torch.tensor([40, 12]))
    sum(losses.values()).backward()
    assert all(torch.isfinite(v) for v in losses.values())
    mel, y_len = m.eval().synthesize(ids, x_len, steps=2)
    assert mel.shape[:2] == (2, 80) and mel.shape[-1] == int(y_len.max())


def test_train_tts_smoke(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    texts, rows, audio, pos = {"train": [], "val": []}, {"train": [], "val": []}, {"train": [], "val": []}, {}
    for split, n in (("train", 6), ("val", 2)):
        pos[split] = 0
        for i in range(n):
            wav = (0.3 * np.sin(np.arange(64 * HOP) * (0.05 + 0.01 * i)) * 32767).astype("<i2")
            audio[split].append(wav)
            rows[split].append((pos[split], len(wav)))
            pos[split] += len(wav)
            texts[split].append([f"utt{i}", "a short test sentence"])
        np.concatenate(audio[split]).tofile(data / f"{split}.bin")
        np.save(data / f"{split}_index.npy", np.array(rows[split], dtype=np.int64))
    (data / "transcripts.json").write_text(json.dumps(texts))
    out = tmp_path / "tts"
    common = [sys.executable, "scripts/train_tts.py", "--small", "--data-dir", str(data), "--out", str(out),
              "--batch-size", "2", "--workers", "0", "--log-every", "1", "--no-amp", "--sample-steps", "2",
              "--vocoder", str(tmp_path / "none.pt")]
    subprocess.run(common + ["--max-steps", "3"], check=True, cwd=ROOT)
    assert (out / "best.pt").exists() and (out / "resume.pt").exists()
    subprocess.run(common + ["--max-steps", "5", "--resume"], check=True, cwd=ROOT)
    assert torch.load(out / "latest.pt", weights_only=False)["step"] == 5
