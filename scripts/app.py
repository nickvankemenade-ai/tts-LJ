"""Local web app: text to speech, plus recording, loading or painting a mel spectrogram to vocode.

  python scripts/app.py          # http://localhost:8765

Uses runs/tts_full/latest.pt (text to mel) and runs/gan_v1/latest.pt (vocoder) by default.
Both are reloaded automatically when training saves a new checkpoint, so the app can be used
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

    def __init__(self, ckpt: Path, device: torch.device, loader):
        self.ckpt, self.device, self.loader = ckpt, device, loader
        self.lock = threading.Lock()
        self.mtime = self.model = None
        self.info = {}

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
            return {"checkpoint": str(self.ckpt), "loaded": self.model is not None, **self.info}

    def get(self):
        self.refresh()
        if self.model is None:
            raise RuntimeError(f"no checkpoint yet at {self.ckpt}")
        return self.model


def mel_payload(mel: torch.Tensor) -> dict:
    return {"frames": mel.shape[1], "n_mels": mel.shape[0], "mel": [round(v, 3) for v in mel.flatten().tolist()]}


def mel_from_payload(body: dict) -> torch.Tensor:
    frames = int(body["frames"])
    if not 1 <= frames <= MAX_FRAMES:
        raise ValueError(f"spectrogram must be 1..{MAX_FRAMES} frames (20 s)")
    return torch.tensor(body["mel"], dtype=torch.float32).reshape(N_MELS, frames)


def make_handler(voc: Reloading, tts: Reloading, melfn: LogMel, data_dir: Path):
    device = voc.device

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
                    self.send_json({"vocoder": voc.status(), "tts": tts.status()})
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
                    with tts.lock:
                        model = tts.get()
                        mel, _ = model.synthesize(torch.tensor([ids], device=device),
                                                  torch.tensor([len(ids)], device=device),
                                                  length_scale=1.0 / speed)
                    self.send_json(mel_payload(mel[0, :, :MAX_FRAMES].float().cpu()))
                elif url.path == "/api/vocode":
                    mel = mel_from_payload(json.loads(self.body()))
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
    runs = DEFAULT_WORK_DIR / "runs"
    p.add_argument("--vocoder", type=Path, default=runs / "gan_v1" / "latest.pt")
    p.add_argument("--tts", type=Path, default=runs / "tts_full" / "latest.pt")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_WORK_DIR / "data", help="for the example recordings")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    voc = Reloading(args.vocoder, device, load_vocoder)
    tts = Reloading(args.tts, device, load_tts)
    print(json.dumps({"vocoder": voc.status(), "tts": tts.status()}))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(voc, tts, LogMel().to(device), args.data_dir))
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
