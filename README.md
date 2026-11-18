# tts-LJ

A text-to-speech system trained from scratch on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), on a single laptop GPU (RTX 4060, 8 GB).

```
text ──► text to mel ──► 80-band log-mel spectrogram ──► vocoder ──► 16 kHz audio
         (flow matching, 15.2M)                          (HiFi-GAN, 13.9M)
```

It includes a local web app for typing text and hearing it, and for recording, loading or painting a spectrogram and hearing what the vocoder makes of it.

## Models

| Stage | Architecture | Parameters |
|---|---|---|
| Text to mel | Matcha-TTS style: transformer encoder, monotonic alignment search, flow-matching decoder | 15.2M |
| Mel to audio | HiFi-GAN V1 generator, multi-period + multi-resolution discriminators | 13.9M |

Both models are trained from random initialization: no pretrained weights, no external aligner, no phonemizer.

### Pretrained models

Two trained checkpoints are included in `pretrained/`:

- **`pretrained/vocoder.pt`** (56 MB): trained on a 10-hour subset of LJSpeech (`prepare_data.py --max-hours 10`).
- **`pretrained/tts.pt`** (61 MB): text to mel, trained on the same 10-hour subset first, then continued on the full dataset (about 24 hours). Training was stopped early, once the validation losses stopped improving. That means fewer training steps, not less data: every batch is drawn from the whole training split, so the model saw all of it many times over.

The text-to-mel model reads characters directly. It learns how spelling maps to sound from the data alone, so it handles most common irregular words ("though", "through", "thought"). It doesn't reliably handle homographs or rare spellings, though (see [Limitations](#limitations)).

## Quick start

Requirements: Python 3.10, PyTorch 2.3 with CUDA, torchaudio and numba. A CPU works for inference, but is slow for training.

```bash
pip install -r requirements.txt
```

The app uses the pretrained models by default, so it runs straight after cloning:

```bash
python scripts/app.py          # opens http://localhost:8765
```

- **Models:** choose **Pretrained**, the checkpoints in `pretrained/`, or **Your own (runs/)**, the checkpoints that `train_gan.py` and `train_tts.py` save.
  - The line under the controls shows which models are in use. If a model is missing from `runs/`, it says so in red, and that stage can't be used until you train it.
- **Text to speech:** type text, set the speed, press **Speak**, then play or download the WAV.
- **Spectrogram:** shows the mel spectrogram the vocoder receives. You can also record from the microphone, load a held-out LJSpeech recording (**Example**), or start from a **Blank** canvas.
- **Paint and erase:** draw on the spectrogram and press **Vocode** to hear the result. With **harmonics** on, the brush also paints the overtones of the pitch you draw, so low lines sound voice-like rather than like pure tones.

With **Your own** selected, the app reloads a checkpoint whenever training saves a new one, so you can listen to a model while it trains.

## Data

`scripts/prepare_data.py` builds `data/` from one of three sources:
- `--download`: LJSpeech 1.1, streamed and decompressed on the fly. The 2.6 GB archive is never written to disk.
- `--archive PATH`: an LJSpeech-style `.tar.bz2` or `.tar.gz` you already have.
- `--wav-dir PATH`: a folder of 16-bit WAV files, any sample rate.

Audio is resampled to 16 kHz and stored once as int16, about 115 MB per hour: `train.bin` and `val.bin`, plus an index. Mel spectrograms are computed on the GPU during training.

**Splits.**
- **Validation:** a clip goes to validation if the MD5 hash of its file name falls in the lowest `--val-percent` (default 1%), until there are `--max-val` (default 100) validation clips.
- **Training:** every other clip.
- **Stable:** the split depends only on file names, so it's the same on every run. For the full LJSpeech that gives 100 validation and 13,000 training clips (23.7 h).

**Capping.** `--max-hours` caps the training audio. Clips are taken in archive order until the cap is reached.
- With `--download` or `--archive`, preparation stops there, so the validation split only holds the validation clips that came before the cap.
- With `--wav-dir`, it keeps reading to fill the validation split.

**Transcripts.** For archives, the normalized text from `metadata.csv` is saved to `transcripts.json`, in the same order as the clips. `--wav-dir` gives audio only, which is enough for the vocoder but not for text to mel.

**How each script uses the splits:**

| Script | Training | Validation |
|---|---|---|
| `train_gan.py` | random 0.5 s segments (`--segment 8192` samples) from training clips at least that long | the first 20 validation clips (`--val-clips`), cut to 4 s; score: log-mel L1 between each clip and its resynthesis |
| `train_tts.py` | training clips up to 12 s that have at least as many mel frames as text tokens | validation clips that pass the same length filter; scores: prior, duration and flow-matching losses, with a fixed noise seed |
| `app.py` | | the **Example** button plays validation clips, cut to 6 s |

## Training

```bash
python scripts/prepare_data.py --download     # full LJSpeech, ~2.7 GB in data/
python scripts/train_gan.py --hours 6         # vocoder, saves to runs/gan/
python scripts/train_tts.py --hours 2         # text to mel, saves to runs/tts/
```

The two models train independently and can run one after the other or in parallel. `train_tts.py` uses a vocoder only to turn its evaluation samples into audio: `runs/gan/latest.pt` if you've trained one, otherwise `pretrained/vocoder.pt`.

- **Budgets:** training runs under a **wall-clock budget**. `--hours` counts the total training time of a run, including time from `--init` or `--resume`, and the learning rate follows a cosine schedule over it.
- **`--init PATH`:** starts text-to-mel training from another checkpoint's weights, with a fresh optimizer.
- **Checkpoints:** every `--eval-minutes`, each script scores held-out clips, writes audio samples to `runs/<name>/samples/`, and saves weights-only checkpoints: `latest.pt`, and `best.pt` for the best validation score.
- **Resuming:** the full training state goes to `resume.pt` (hourly for the vocoder, every 20 minutes for text to mel), and `--resume` continues an interrupted run.

Measured on an RTX 4060 Laptop GPU with bf16 autocast:

| Run | Batch | Steps/s | GPU memory |
|---|---|---|---|
| Vocoder, mel-only warm-up | 16 × 0.5 s | ~70 | 1.4 GB |
| Vocoder, adversarial | 16 × 0.5 s | ~2.7–3.1 | 3.4 GB |
| Text to mel | 32 utterances | ~8.3 | 1.9 GB |

## How it works

### Text to mel

The model follows [Matcha-TTS](https://arxiv.org/abs/2309.03199):

- **Input:** lowercase characters and punctuation, with a blank token between every pair of characters, as in Glow-TTS.
- **Encoder:** a convolutional prenet, then 6 transformer blocks (192 channels, rotary position embeddings, convolutional feed-forward). For every token it predicts a mean mel frame and a duration.
- **Alignment:** at every training step, [monotonic alignment search](https://arxiv.org/abs/2005.11129) finds the most likely alignment between tokens and the target mel frames. It runs in numba on the CPU. The duration predictor learns the resulting durations, so no external aligner is needed.
- **Decoder:** conditional flow matching. It learns the velocity along a straight path from noise to the mel, conditioned on the encoder's means expanded to frame rate. It has residual convolution blocks and transformer blocks (256 channels), and trains on random 2-second crops.
- **Losses:** prior (mel against the expanded means), duration, and flow matching.
- **Inference:** 20 Euler steps from noise at temperature 0.3.

### Vocoder

The vocoder follows [HiFi-GAN](https://arxiv.org/abs/2010.05646):

- **Generator:** the V1 configuration. It upsamples by 8 × 8 × 2 × 2 = 256, the mel hop size, and uses multi-receptive-field residual blocks.
- **Discriminators:** the five HiFi-GAN period discriminators at half width, plus three [UnivNet](https://arxiv.org/abs/2106.07889)-style multi-resolution STFT discriminators (FFT sizes 512, 1024 and 2048). They take about half the compute of the original multi-scale set.
- **Losses:** least-squares adversarial loss, feature matching, and mel L1 (weight 45).
- **Warm-up:** the first 20 minutes train on mel L1 only, which makes output intelligible quickly before the adversarial phase.
- **Training tricks:** EMA weights (0.999), bf16 autocast, and weight norm throughout. PyTorch's spectral-norm parametrization fails under bf16.

### Audio features

- 16 kHz audio, 1024-point FFT, hop 256 (16 ms), 80 Slaney mel bands from 0 to 8 kHz.
- Natural-log magnitudes, clamped at 1e-5.
- Frames are exactly `samples // 256`.

## Limitations

- **One voice.** LJSpeech is a single female speaker reading non-fiction, so that's the only voice and style.
- **Characters only.** There is no text normalization: digits are dropped and abbreviations are read as written, so write "forty two" and "Doctor". Other languages are read with English spelling rules.
- **Homographs.** The model mostly doesn't use context to choose a pronunciation. It reads "read" as "red" except directly after "to", and always reads "lead" as "leed". LJSpeech has only a few dozen examples of each, and the model learned the simplest pattern they support.
- **Rare words.** Words seen once or never ("tough", "plough") are often mispronounced.
- **Length.** The app caps output at 20 seconds.

## Repository layout

```
tms/            model code
  tts.py          text to mel: text processing, encoder, alignment search, flow-matching decoder
  gan.py          vocoder: generator, discriminators, losses
  audio.py        log-mel features, WAV input/output, resampling
  data.py         memory-mapped dataset, length-bucketed batches
  common.py       checkpoint loading and shared helpers
  app/index.html  the web app's page
scripts/
  prepare_data.py  download LJSpeech (or read a WAV folder) into data/
  train_gan.py     train the vocoder
  train_tts.py     train the text-to-mel model
  app.py           the web app server (no web framework needed)
pretrained/      the released checkpoints: tts.pt (text to mel) and vocoder.pt
runs/            your own training runs (created by the training scripts, not in git)
tests/           shapes, alignment search, data preparation, short training runs
```

```bash
python -m pytest -q tests
```

## References

- Mehta et al., [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://arxiv.org/abs/2309.03199), 2024.
- Kim et al., [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search](https://arxiv.org/abs/2005.11129), 2020.
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), 2023.
- Kong et al., [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646), 2020.
- Jang et al., [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators](https://arxiv.org/abs/2106.07889), 2021.
- Ito & Johnson, [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/), 2017 (public domain).

## License

The code is released under the [MIT License](LICENSE). LJSpeech, which the models are trained on, is in the public domain.
