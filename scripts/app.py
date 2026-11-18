"""Local web app: text to speech, plus recording, loading or painting a mel spectrogram to vocode.

  python scripts/app.py          # http://localhost:8765

Two model sets, chosen per request with the page's Models menu:
  pretrained   pretrained/tts.pt and pretrained/vocoder.pt (the released checkpoints)
  own          runs/tts/latest.pt and runs/gan/latest.pt (what train_tts.py / train_gan.py save)
Own checkpoints are reloaded automatically when training saves a new one, so the app can be used
while a model trains. Uses only the standard library plus the repo's own dependencies.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tms.audio import (FLOOR, HOP, N_MELS, SAMPLE_RATE, LogMel, bin_frequencies,  # noqa: E402
                           to_model_audio, write_wav)
from tms.common import DEFAULT_WORK_DIR, load_vocoder, vocode  # noqa: E402
from tms.data import load_utterances  # noqa: E402
from tms.tts import load_tts, text_to_ids  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "tms" / "app" / "index.html"
MAX_FRAMES = 20 * SAMPLE_RATE // HOP  # 20 seconds


class Reloading:
    """Holds a model and reloads it when its checkpoint file changes."""

    def __init__(self, ckpt: Path, device: torch.device, loader, label: str = "model"):
        self.ckpt, self.device, self.loader, self.label = ckpt, device, loader, label
        self.lock = threading.Lock()
        self.mtime = self.model = None
        self.info = {}
        try:  # folder shown to the user, relative to the repo where possible, e.g. "runs/tts/"
            self.dir = self.ckpt.resolve().parent.relative_to(DEFAULT_WORK_DIR).as_posix() + "/"
        except ValueError:
            self.dir = str(self.ckpt.parent)

    def refresh(self) -> None:
        if not self.ckpt.exists():
            return
        mtime = self.ckpt.stat().st_mtime
        if mtime != self.mtime:
            self.model, ck = self.loader(self.ckpt, self.device)
            self.mtime = mtime
            self.info = {"step": ck["step"], "hours": round(ck["elapsed"] / 3600, 2)}
            print(f"loaded {self.ckpt} (step {ck['step']})")

    def status(self) -> dict:
        with self.lock:
            self.refresh()
            return {"dir": self.dir, "loaded": self.model is not None, **self.info}

    def get(self):
        self.refresh()
        if self.model is None:
            raise RuntimeError(f"No {self.label} model found in {self.dir}")
        return self.model


def mel_payload(mel: torch.Tensor) -> dict:
    return {"frames": mel.shape[1], "n_mels": mel.shape[0], "mel": [round(v, 3) for v in mel.flatten().tolist()]}


def mel_from_payload(body: dict) -> torch.Tensor:
    frames = int(body["frames"])
    if not 1 <= frames <= MAX_FRAMES:
        raise ValueError(f"spectrogram must be 1..{MAX_FRAMES} frames (20 s)")
    return torch.tensor(body["mel"], dtype=torch.float32).reshape(N_MELS, frames)


class ModelSets:
    """The pretrained and the user's own checkpoints for both stages."""

    def __init__(self, paths: dict[str, dict[str, Path]], device: torch.device):
        loaders = {"tts": load_tts, "vocoder": load_vocoder}
        labels = {"tts": "text-to-mel", "vocoder": "vocoder"}
        self.device = device
        self.sets = {name: {stage: Reloading(path, device, loaders[stage], labels[stage])
                            for stage, path in stages.items()}
                     for name, stages in paths.items()}

    def pick(self, which: str, stage: str) -> Reloading:
        """The model for ``stage`` ("tts" or "vocoder") in set ``which`` ("pretrained" or "own")."""
        if which not in self.sets:
            raise ValueError(f"unknown model set {which!r}")
        return self.sets[which][stage]

    def status(self) -> dict:
        return {name: {stage: r.status() for stage, r in stages.items()} for name, stages in self.sets.items()}


def make_handler(models: ModelSets, melfn: LogMel, data_dir: Path):
    device = models.device

    def wav_bytes(wav: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        write_wav(buf, wav)
        return buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter console
            pass

        def send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, code: int = 200):
            self.send(code, json.dumps(obj).encode(), "application/json")

        def body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/info":
                    self.send_json({"sample_rate": SAMPLE_RATE, "hop": HOP, "n_mels": N_MELS, "floor": FLOOR,
                                    "bins_hz": bin_frequencies(), "max_frames": MAX_FRAMES})
                elif url.path == "/api/status":
                    self.send_json(models.status())
                elif url.path in ("/api/example", "/api/example_wav"):
                    i = int(parse_qs(url.query).get("i", ["0"])[0])
                    utts = load_utterances(data_dir, "val", n=i + 1, max_seconds=6.0)
                    wav = utts[i % len(utts)]
                    if url.path == "/api/example":
                        self.send_json(mel_payload(melfn(wav.to(device)[None])[0].cpu()))
                    else:  # the original recording, for comparison
                        self.send(200, wav_bytes(wav), "audio/wav")
                else:
                    self.send(404, b"not found", "text/plain")
            except Exception as e:  # report errors to the page
                self.send_json({"error": str(e)}, 500)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                if url.path == "/api/mel":  # raw little-endian float32 audio
                    sr = int(parse_qs(url.query).get("sr", [str(SAMPLE_RATE)])[0])
                    wav = to_model_audio(np.frombuffer(self.body(), dtype="<f4"), sr)[: MAX_FRAMES * HOP]
                    if len(wav) < HOP * 4:
                        raise ValueError("recording too short")
                    self.send_json(mel_payload(melfn(wav.to(device)[None])[0].cpu()))
                elif url.path == "/api/tts":
                    body = json.loads(self.body())
                    ids = text_to_ids(str(body.get("text", ""))[:1000])
                    if len(ids) < 3:
                        raise ValueError("type some text (letters and punctuation)")
                    speed = min(2.0, max(0.5, float(body.get("speed", 1.0))))
                    tts = models.pick(str(body.get("models", "pretrained")), "tts")
                    with tts.lock:
                        model = tts.get()
                        mel, _ = model.synthesize(torch.tensor([ids], device=device),
                                                  torch.tensor([len(ids)], device=device),
                                                  length_scale=1.0 / speed)
                    self.send_json(mel_payload(mel[0, :, :MAX_FRAMES].float().cpu()))
                elif url.path == "/api/vocode":
                    body = json.loads(self.body())
                    mel = mel_from_payload(body)
                    voc = models.pick(str(body.get("models", "pretrained")), "vocoder")
                    with voc.lock:
                        wav = vocode(voc.get(), mel.to(device)[None])[0].cpu()
                    self.send(200, wav_bytes(wav), "audio/wav")
                else:
                    self.send(404, b"not found", "text/plain")
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained-dir", type=Path, default=DEFAULT_WORK_DIR / "pretrained")
    p.add_argument("--own-vocoder", type=Path, default=DEFAULT_WORK_DIR / "runs" / "gan" / "latest.pt")
    p.add_argument("--own-tts", type=Path, default=DEFAULT_WORK_DIR / "runs" / "tts" / "latest.pt")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_WORK_DIR / "data", help="for the example recordings")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    models = ModelSets({
        "pretrained": {"tts": args.pretrained_dir / "tts.pt", "vocoder": args.pretrained_dir / "vocoder.pt"},
        "own": {"tts": args.own_tts, "vocoder": args.own_vocoder},
    }, device)
    print(json.dumps(models.status()))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(models, LogMel().to(device), args.data_dir))
    url = f"http://localhost:{args.port}"
    print(f"app running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
