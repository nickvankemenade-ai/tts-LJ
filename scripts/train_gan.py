"""Train the HiFi-GAN-style vocoder from scratch under a wall-clock budget.

  python scripts/train_gan.py --hours 6            (v1 generator, light discriminators)

Every --eval-minutes it vocodes held-out utterances, logs the log-mel distance, writes audio
samples and saves small weights-only checkpoints:
  latest.pt   the most recent weights (EMA weights when --ema > 0)
  best.pt     the best val score among adversarial-phase checkpoints
  resume.pt   full training state (discriminators, optimizers), every --resume-minutes and at the end
Rerun the same command with --resume to continue; the time budget counts the earlier training
time. Add --synthetic to benchmark speed and memory without a dataset.
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

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tms.audio import SAMPLE_RATE, LogMel, write_wav  # noqa: E402
from tms.common import DEFAULT_WORK_DIR, count_params, mel_distance, vocode  # noqa: E402
from tms.data import SegmentDataset, load_utterances  # noqa: E402
from tms.gan import (GENERATOR_PRESETS, Generator, GeneratorConfig, MultiDiscriminator,  # noqa: E402
                         discriminator_loss, feature_matching_loss, generator_adv_loss)


def synthetic_batches(batch_size: int, segment: int, device):
    """Harmonic tones plus noise: only for speed/memory benchmarks and smoke tests."""
    t = torch.arange(segment, device=device) / SAMPLE_RATE
    while True:
        f0 = torch.rand(batch_size, 1, device=device) * 200 + 100
        wav = sum(torch.sin(2 * math.pi * f0 * k * t) / k for k in range(1, 6)) * 0.2
        yield wav + 0.01 * torch.randn_like(wav)


def real_batches(args, device):
    ds = SegmentDataset(args.data_dir, "train", segment=args.segment)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    while True:
        for batch in dl:
            yield batch.to(device, non_blocking=True)


def atomic_save(obj, path: Path) -> None:
    """Write to a temp file, then swap it in, so readers never see a half-written checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=6.0, help="wall-clock training budget")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_WORK_DIR / "data")
    p.add_argument("--out", type=Path, default=DEFAULT_WORK_DIR / "runs" / "gan_v1")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--segment", type=int, default=8192, help="samples per training clip")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-min-frac", type=float, default=0.1,
                   help="cosine decay over the time budget down to lr * this (1 = constant lr)")
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay for saved/evaluated weights (0 = off)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    p.add_argument("--eval-minutes", type=float, default=20.0)
    p.add_argument("--val-clips", type=int, default=20)
    p.add_argument("--resume-minutes", type=float, default=60.0, help="how often to write the full resume.pt")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="random harmonic audio, no dataset needed")
    p.add_argument("--small", action="store_true", help="tiny models, for tests")
    p.add_argument("--gen", choices=sorted(GENERATOR_PRESETS), default="v1", help="generator preset")
    p.add_argument("--disc", choices=["light", "hifigan"], default="light", help="discriminator preset")
    p.add_argument("--mel-only-minutes", type=float, default=20.0,
                   help="train the generator on mel L1 only for this long before switching on the discriminators")
    p.add_argument("--mel-weight", type=float, default=45.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = not args.no_amp and device.type == "cuda"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    config = dict(GENERATOR_PRESETS[args.gen])
    if args.small:
        config["upsample_initial_channel"] = 32
    model = Generator(GeneratorConfig(**config)).to(device)
    ema = None
    if args.ema > 0:
        ema = copy.deepcopy(model).eval()
        for prm in ema.parameters():
            prm.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    disc = MultiDiscriminator(args.disc).to(device)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99))
    melfn = LogMel().to(device)

    step, elapsed, best = 0, 0.0, float("inf")
    if args.resume:
        ck = torch.load(out / "resume.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["train_model"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        disc.load_state_dict(ck["disc"])
        opt_d.load_state_dict(ck["opt_d"])
        step, elapsed, best = ck["step"], ck["elapsed"], ck.get("best", best)
        print(f"resumed at step {step}, {elapsed / 3600:.2f} h already trained")

    print(f"gan ({args.gen} generator, {args.disc} discriminators): {count_params(model) / 1e6:.2f}M params, "
          f"discriminator {count_params(disc) / 1e6:.2f}M | device {device} | bf16 {amp} | "
          f"batch {args.batch_size} x {args.segment} samples | lr {args.lr} cosine to x{args.lr_min_frac} | ema {args.ema}")

    if args.synthetic:
        batches = synthetic_batches(args.batch_size, args.segment, device)
        val = [next(synthetic_batches(1, 4 * 8192, device))[0].cpu() for _ in range(2)]
    else:
        batches = real_batches(args, device)
        val = load_utterances(args.data_dir, "val", n=args.val_clips)
        if not val:
            raise SystemExit("validation split is empty: rerun prepare_data.py with a larger --val-percent")
    (out / "reference").mkdir(exist_ok=True)
    for i, wav in enumerate(val[:3]):
        write_wav(out / "reference" / f"val{i}.wav", wav)

    log_f = open(out / "log.jsonl", "a")

    def log(rec):
        rec.update(step=step, hours=round(elapsed / 3600, 4))
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    def eval_model():
        return ema if ema is not None else model

    def evaluate():
        net = eval_model()
        net.eval()
        dists = []
        for i, wav in enumerate(val):
            ref = wav.to(device)[None]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                pred = vocode(net, melfn(ref)).float()
            dists.append(mel_distance(melfn, pred, ref))
            if i < 3:
                write_wav(out / "samples" / f"step{step:07d}_val{i}.wav", pred[0])
        model.train()
        return sum(dists) / len(dists)

    def weights(val_dist):
        return {"kind": "gan", "config": model.cfg.to_dict(), "model": eval_model().state_dict(),
                "step": step, "elapsed": elapsed, "best": best, "val_mel_dist": val_dist, "ema_decay": args.ema}

    def checkpoint(val_dist, adversarial: bool, final: bool = False):
        nonlocal best, last_resume
        atomic_save(weights(val_dist), out / "latest.pt")
        # mel-only checkpoints score well on log-mel distance but sound robotic
        if adversarial and val_dist < best:
            best = val_dist
            atomic_save(weights(val_dist), out / "best.pt")
        if final or elapsed - last_resume >= args.resume_minutes * 60:
            last_resume = elapsed
            ck = weights(val_dist)
            ck.update(train_model=model.state_dict(), opt=opt.state_dict(),
                      disc=disc.state_dict(), opt_d=opt_d.state_dict())
            if ema is not None:
                ck["ema"] = ema.state_dict()
            atomic_save(ck, out / "resume.pt")

    budget = args.hours * 3600
    last_eval = last_resume = elapsed
    t_last, steps_since = time.time(), 0
    run_start = time.time() - elapsed
    acc = {}
    model.train()
    adversarial = False
    while elapsed < budget and (args.max_steps is None or step < args.max_steps):
        # cosine learning-rate decay over the wall-clock budget
        frac = min(1.0, elapsed / budget)
        lr = args.lr * (args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
        for o in (opt, opt_d):
            for g in o.param_groups:
                g["lr"] = lr

        wav = next(batches).unsqueeze(1)  # (B, 1, T)
        mel = melfn(wav)
        autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)
        with autocast:
            fake = model(mel)
        loss_mel = F.l1_loss(melfn(fake), mel)
        parts = {"mel": loss_mel.item()}
        adversarial = elapsed >= args.mel_only_minutes * 60
        if adversarial:
            # discriminator step
            with autocast:
                outs_r, outs_f, _, _ = disc(wav, fake.detach())
            loss_d = discriminator_loss(outs_r, outs_f)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            # generator adversarial + feature-matching terms
            with autocast:
                _, outs_f, fm_r, fm_f = disc(wav, fake)
            loss_adv = generator_adv_loss(outs_f)
            loss_fm = feature_matching_loss(fm_r, fm_f)
            loss = loss_adv + loss_fm + args.mel_weight * loss_mel
            parts.update(d=loss_d.item(), adv=loss_adv.item(), fm=loss_fm.item())
        else:
            loss = args.mel_weight * loss_mel
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - args.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)

        step += 1
        steps_since += 1
        elapsed = time.time() - run_start
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v

        if step % args.log_every == 0:
            now = time.time()
            rec = {k: round(v / steps_since, 5) for k, v in acc.items()}
            rec["steps_per_s"] = round(steps_since / (now - t_last), 2)
            rec["lr"] = float(f"{lr:.3g}")
            if device.type == "cuda":
                rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print(f"step {step:>7} | {elapsed / 60:6.1f} min | " + " ".join(f"{k} {v}" for k, v in rec.items()))
            log(rec)
            t_last, steps_since, acc = now, 0, {}

        if elapsed - last_eval >= args.eval_minutes * 60:
            last_eval = elapsed
            dist = evaluate()
            print(f"  eval: val log-mel distance {dist:.4f}")
            log({"val_mel_dist": dist, "adversarial": adversarial})
            checkpoint(dist, adversarial)

    dist = evaluate()
    print(f"done: {step} steps in {elapsed / 3600:.2f} h, val log-mel distance {dist:.4f}")
    log({"val_mel_dist": dist, "adversarial": adversarial, "final": True})
    checkpoint(dist, adversarial, final=True)
    log_f.close()


if __name__ == "__main__":
    main()
