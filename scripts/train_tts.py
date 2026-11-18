"""Train the Matcha-style text-to-mel model from scratch under a wall-clock budget.

  python scripts/train_tts.py --hours 6                 (base preset)
  python scripts/train_tts.py --hours 6 --preset small

Needs data/transcripts.json (written by prepare_data.py). Every --eval-minutes it computes the
validation losses (fixed noise seed), synthesizes a few sentences, renders them with the vocoder
in --vocoder, and saves weights-only checkpoints:
  latest.pt   most recent (EMA) weights
  best.pt     lowest validation loss
  resume.pt   full training state, every --resume-minutes and at the end
Rerun with --resume to continue; the time budget counts the earlier training time.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tms.audio import LogMel, write_wav  # noqa: E402
from tms.common import DEFAULT_WORK_DIR, count_params, load_vocoder, vocode  # noqa: E402
from tms.data import TextAudioDataset, collate_text_audio  # noqa: E402
from tms.tts import TTS_PRESETS, MatchaTTS, TTSConfig, text_to_ids  # noqa: E402

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a test.",
]


class InfiniteLengthBatches(torch.utils.data.Sampler):
    def __init__(self, ds: TextAudioDataset, batch_size: int, seed: int):
        self.ds, self.batch_size, self.seed = ds, batch_size, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            yield from self.ds.length_batches(self.batch_size, rng)


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def mel_stats(ds: TextAudioDataset, melfn, device, n: int = 300) -> tuple[float, float]:
    vals = []
    for i in np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int):
        vals.append(melfn(ds[i][1].to(device)[None]).flatten())
    v = torch.cat(vals)
    return v.mean().item(), v.std().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--preset", choices=sorted(TTS_PRESETS), default="base")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_WORK_DIR / "data")
    p.add_argument("--out", type=Path, default=DEFAULT_WORK_DIR / "runs" / "tts")
    p.add_argument("--vocoder", type=Path, default=None,
                   help="vocoder for the audio samples (default: runs/gan/latest.pt if it exists, "
                        "else pretrained/vocoder.pt; samples are skipped if neither exists)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--lr-min-frac", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--eval-minutes", type=float, default=20.0)
    p.add_argument("--resume-minutes", type=float, default=20.0)
    p.add_argument("--sample-steps", type=int, default=20, help="flow-matching Euler steps for samples")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", type=Path, default=None,
                   help="start from the weights in a latest.pt/best.pt (fresh optimizer), e.g. after a crash "
                        "with no resume.pt; step and elapsed time carry over")
    p.add_argument("--small", action="store_true", help="tiny model, for tests")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = not args.no_amp and device.type == "cuda"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    melfn = LogMel().to(device)

    train_ds = TextAudioDataset(args.data_dir, "train")
    val_ds = TextAudioDataset(args.data_dir, "val")
    config = dict(TTS_PRESETS[args.preset])
    if args.small:
        config.update(enc_channels=32, enc_ffn=64, enc_layers=1, dp_channels=32, dec_channels=32, dec_groups=1,
                      enc_heads=2, dec_heads=2, segment_frames=16)

    step, elapsed, best = 0, 0.0, float("inf")
    ck = None
    if args.resume:
        ck = torch.load(out / "resume.pt", map_location=device, weights_only=False)
        config = ck["config"]
    elif args.init:
        init = torch.load(args.init, map_location=device, weights_only=False)
        config = init["config"]
    else:
        config["mel_mean"], config["mel_std"] = mel_stats(train_ds, melfn, device)
    model = MatchaTTS(TTSConfig(**config)).to(device)
    ema = copy.deepcopy(model).eval() if args.ema > 0 else None
    if ema is not None:
        for prm in ema.parameters():
            prm.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    if ck is not None:
        model.load_state_dict(ck["train_model"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, elapsed, best = ck["step"], ck["elapsed"], ck["best"]
        print(f"resumed at step {step}, {elapsed / 3600:.2f} h already trained")
    elif args.init:
        model.load_state_dict(init["model"])  # EMA weights; the training weights restart from them
        if ema is not None:
            ema.load_state_dict(init["model"])
        step, elapsed = init["step"], init["elapsed"]
        print(f"initialized from {args.init} (step {step}, {elapsed / 3600:.2f} h), fresh optimizer with warm-up")
    start_step = step

    if args.vocoder is None:
        own = DEFAULT_WORK_DIR / "runs" / "gan" / "latest.pt"
        args.vocoder = own if own.exists() else DEFAULT_WORK_DIR / "pretrained" / "vocoder.pt"
    vocoder = None
    if args.vocoder.exists():
        vocoder = load_vocoder(args.vocoder, device)[0]
    print(f"tts ({args.preset}): {count_params(model) / 1e6:.2f}M params "
          f"(encoder {count_params(model.encoder) / 1e6:.2f}M, decoder {count_params(model.decoder) / 1e6:.2f}M)"
          f" | {len(train_ds)} train / {len(val_ds)} val utterances | batch {args.batch_size} | bf16 {amp}"
          f" | lr {args.lr} | mel mean {config['mel_mean']:.2f} std {config['mel_std']:.2f}"
          f" | vocoder {args.vocoder if vocoder else 'none'}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=InfiniteLengthBatches(train_ds, args.batch_size, seed=step),
        collate_fn=collate_text_audio, num_workers=args.workers, pin_memory=False,  # variable shapes: pinned cache grows
        persistent_workers=args.workers > 0)
    batches = iter(loader)
    val_batches = [collate_text_audio([val_ds[j] for j in range(i, min(i + 16, len(val_ds)))])
                   for i in range(0, len(val_ds), 16)]
    log_f = open(out / "log.jsonl", "a")

    def log(rec):
        rec.update(step=step, hours=round(elapsed / 3600, 4))
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)

    def compute_loss(net, x, x_len, w, y_len):
        x, x_len, w, y_len = (a.to(device, non_blocking=True) for a in (x, x_len, w, y_len))
        mel = melfn(w)
        with autocast():
            return net.loss(x, x_len, mel, y_len)

    @torch.no_grad()
    def evaluate():
        net = ema or model
        net.eval()
        tot = {}
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(1234)
            for b in val_batches:
                for k, v in compute_loss(net, *b).items():
                    tot[k] = tot.get(k, 0.0) + v.item() * len(b[1])
            n = sum(len(b[1]) for b in val_batches)
            res = {f"val_{k}": v / n for k, v in tot.items()}
            texts = [val_ds.items[i][3] for i in range(2)] + TEST_SENTENCES
            for i, text in enumerate(texts):
                ids = torch.tensor([text_to_ids(text)], device=device)
                with autocast():
                    mel, _ = net.synthesize(ids, torch.tensor([ids.shape[1]], device=device), steps=args.sample_steps)
                if vocoder is not None:
                    with autocast():
                        wav = vocode(vocoder, mel.float()).float()
                    write_wav(out / "samples" / f"step{step:07d}_{i}.wav", wav[0].clamp(-1, 1))
        model.train()
        res["val_total"] = res["val_prior"] + res["val_dur"] + res["val_fm"]
        return res

    def weights(res):
        return {"kind": "tts", "config": model.cfg.to_dict(), "model": (ema or model).state_dict(),
                "step": step, "elapsed": elapsed, "best": best, **res}

    def checkpoint(res, final=False):
        nonlocal best, last_resume
        atomic_save(weights(res), out / "latest.pt")
        if res["val_total"] < best:
            best = res["val_total"]
            atomic_save(weights(res), out / "best.pt")
        if final or elapsed - last_resume >= args.resume_minutes * 60:
            last_resume = elapsed
            ck = weights(res)
            ck.update(train_model=model.state_dict(), opt=opt.state_dict(), best=best)
            if ema is not None:
                ck["ema"] = ema.state_dict()
            atomic_save(ck, out / "resume.pt")

    budget = args.hours * 3600
    last_eval = last_resume = elapsed
    t_last, steps_since, acc = time.time(), 0, {}
    run_start = time.time() - elapsed
    model.train()
    while elapsed < budget and (args.max_steps is None or step < args.max_steps):
        frac = min(1.0, elapsed / budget)
        lr = args.lr * min(1.0, (step - start_step + 1) / args.warmup_steps if args.init else (step + 1) / args.warmup_steps) * (
            args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr

        losses = compute_loss(model, *next(batches))
        loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - args.ema)

        step += 1
        steps_since += 1
        elapsed = time.time() - run_start
        for k, v in losses.items():
            acc[k] = acc.get(k, 0.0) + v.item()
        acc["gnorm"] = acc.get("gnorm", 0.0) + gnorm.item()

        if step % args.log_every == 0:
            now = time.time()
            rec = {k: round(v / steps_since, 4) for k, v in acc.items()}
            rec["steps_per_s"] = round(steps_since / (now - t_last), 2)
            rec["lr"] = float(f"{lr:.3g}")
            if device.type == "cuda":
                rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print(f"step {step:>7} | {elapsed / 60:6.1f} min | " + " ".join(f"{k} {v}" for k, v in rec.items()),
                  flush=True)
            log(rec)
            t_last, steps_since, acc = now, 0, {}

        if elapsed - last_eval >= args.eval_minutes * 60:
            last_eval = elapsed
            res = evaluate()
            print("  eval: " + " ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
            log(dict(res))
            checkpoint(res)

    res = evaluate()
    print(f"done: {step} steps in {elapsed / 3600:.2f} h | " + " ".join(f"{k} {v:.4f}" for k, v in res.items()))
    log({**res, "final": True})
    checkpoint(res, final=True)
    log_f.close()


if __name__ == "__main__":
    main()
