# tts-LJ

A text-to-speech system trained **from scratch** on a laptop GPU (RTX 4060, 8 GB), in two models:

- **Text to mel:** a Matcha-TTS-style model (15.2M params).
  - Input is characters, with a blank token between each.
  - A transformer encoder with RoPE predicts a mean mel and a duration per token.
  - The alignment between text and audio is learned with monotonic alignment search, so no external aligner is needed.
  - A flow-matching decoder turns noise (temperature 0.3) into the mel in 20 Euler steps.
- **Mel to audio:** a HiFi-GAN vocoder.
  - Generator: v1 preset, 13.9M params.
  - Discriminators: half-width multi-period plus multi-resolution STFT.

Both work on 16 kHz audio with 80-band log-mel spectrograms (hop 256, FFT 1024).

## The models

| Model | Checkpoint | Data | Training | Final step |
|---|---|---|---|---|
| Vocoder | `runs/gan_v1/latest.pt` | LJSpeech, 10 h | 6 h | 67,106 (val log-mel distance 0.208) |
| Text to mel | `runs/tts_full/latest.pt` | LJSpeech, 10 h, then all 24 h | 0.8 h + 1.2 h | 61,007 |

`data/` and `runs/` are gitignored.

## Web app

```bash
conda activate torch-gpu
python scripts/app.py          # http://localhost:8765
```

- **Text to speech:** type text, set the speed, and press Speak. You can play and download the result.
- **Spectrogram:** shows the mel the vocoder hears. It can come from a microphone recording, a held-out LJSpeech recording, or a blank canvas.
- **Painting:** paint or erase on the spectrogram. "Harmonics" also paints the overtones of the pitch you draw. Then press Vocode.

The app reloads a checkpoint whenever training saves a new one, so it can be used during training.

## Reproducing the models

Use the `torch-gpu` conda environment (PyTorch 2.3 + CUDA 12.1, torchaudio, numba; see `requirements.txt`).

`prepare_data.py --download` streams LJSpeech and decompresses it on the fly; the 2.6 GB archive is never saved to disk. Audio is stored once as 16 kHz int16, about 115 MB per hour, together with the transcripts.

```bash
# 1. the 10 h subset (~1.2 GB); both models start here
python scripts/prepare_data.py --download --max-hours 10 --data-dir data_10h

# 2. vocoder: 6 h (20 min mel-only warm-up, then adversarial training)
python scripts/train_gan.py --hours 6 --data-dir data_10h --out runs/gan_v1

# 3. text to mel, first stage on 10 h
python scripts/train_tts.py --hours 0.83 --eval-minutes 10 --workers 1 --data-dir data_10h --out runs/tts_10h

# 4. the full 24 h (~2.7 GB), then continue the text-to-mel model on it
python scripts/prepare_data.py --download
python scripts/train_tts.py --init runs/tts_10h/latest.pt --hours 2.83 --eval-minutes 10 --workers 1 --out runs/tts_full
```

- **`--hours` is a total:** it includes the time already trained, and the learning rate follows a cosine curve over it.
- **`--init`** continues from a checkpoint's weights with a fresh optimizer.
- **The released checkpoints:**
  - The first text-to-mel stage stopped at 50 minutes, step 25,279 (it crashed once, and the run was continued with `--init`).
  - The second stage was stopped by hand at 2.0 h, step 61,007, once validation losses had stopped improving.
  - The vocoder ran its full 6 h.
- **Resuming:** `--resume` continues an interrupted run from its `resume.pt`. The vocoder writes `resume.pt` hourly, the text-to-mel model every 20 minutes.

**Measured speed:**

| Model | Batch | Steps/s | GPU memory |
|---|---|---|---|
| Vocoder, mel-only warm-up | 16 | ~70 | 1.4 GB |
| Vocoder, adversarial | 16 | ~2.7–3.1 | 3.4 GB |
| Text to mel (base) | 32 | ~8.3 | 1.9 GB |

## Design notes

- **Vocoder warm-up.** For the first 20 minutes (`--mel-only-minutes`), the generator learns from mel L1 only. That makes output intelligible quickly; then the discriminators switch on.
  - `best.pt` only considers adversarial-phase checkpoints, because mel-only checkpoints score well but sound robotic.
- **EMA and learning rate.** Saved weights are an exponential moving average (`--ema 0.999`). The learning rate decays on a cosine curve over the time budget, down to 10%.
- **Weight norm everywhere.** PyTorch's spectral-norm parametrization fails under bf16 autocast, so the discriminators use weight norm.
- **Text-to-mel training.**
  - Batches are grouped by length, so little time goes on padding.
  - The decoder trains on random 2-second crops.
  - Pinned memory is off, because variable batch shapes made the pinned-memory cache grow until Windows ran out of commit memory.
- **Characters, not phonemes.** The model learns spelling-to-sound from data. It handles common irregular words, but not every homograph.

## Tests

```bash
python -m pytest -q tests
```

The tests cover:
- tensor shapes;
- the alignment search, checked against brute force;
- data preparation;
- training, saving and resuming both models for a few steps on a tiny synthetic dataset.
