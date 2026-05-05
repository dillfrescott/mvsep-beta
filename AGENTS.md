# AGENTS.md

## What this is

PyTorch vocal/instrumental source separation model ("NeuralModel"). Single-file architecture + training loop in `main/mvsep.py`. No tests, no lint, no CI, no typecheck — this is a research/experiment repo in active flux.

## Structure

```
main/
  mvsep.py              # ALL model code, dataset, training, inference — the only real source file
  requirements.txt      # torch, torchaudio, torchcodec, adam-atan2-pytorch, soundfile, tqdm
  tools/
    batch_process.py    # batch inference over a directory of audio files
    multisong_eval.py   # evaluation script expecting *_mixture.wav naming convention
```

## Environment

All testing and running should use the **`sep`** conda environment — it has the correct PyTorch and library versions pre-installed.

```
conda run -n sep <command>
```

## Commands

Install deps (from `main/`):
```
pip install -r requirements.txt
```

Train (from `main/`):
```
python mvsep.py --train --data_dir <train_dir> --test_dir <test_dir>
```

Infer (from `main/`):
```
python mvsep.py --infer --input_file <file.wav> [--checkpoint_path <path>] [--latest]
```

Batch infer:
```
python tools/batch_process.py --infer --checkpoint_path <path> --input_dir <dir> [--output_dir <dir>]
```

Multisong eval (expects `<id>_mixture.wav` in input dir, ground truth vocals/other in same dir):
```
python tools/multisong_eval.py --infer --checkpoint_path <path> --input_dir <dir>
```

## Key constraints

- **No code comments**: Do not add comments to any `.py` file.

- **Audio format**: 44100 Hz, stereo (mono auto-upmixed). Inference rejects non-44100 input.
- **Dataset layout** (training): each track is a subdirectory containing `vocals.{wav,flac}` and `other.{wav,flac}`. Both stems required.
- **Checkpoint auto-selection**: `--infer` without `--checkpoint_path` picks best SDR from `best_ckpts/` by default, or latest from `ckpts/` with `--latest`.
- **Checkpoint dirs**: `ckpts/` (regular, max 3 kept) and `best_ckpts/` (only best SDR kept — old bests deleted on new best). Created automatically.
- **Segment length**: 264600 samples (~6 seconds at 44.1kHz) used for both training chunks and inference overlap-add.
- **STFT params**: n_fft=4096, hop_length=1024, window_size=4096 — hardcoded in multiple places, not centralized.
- **EMA**: Exponential moving average (decay 0.999) applied to model weights during training. Checkpoint payload includes both `model_state_dict` and `ema_state_dict`.
- **Optimizer**: `AdamAtan2` from `adam-atan2-pytorch`, lr=1e-4.
- **Gradient checkpointing**: `--ckpt` flag enables it to reduce VRAM at training time.

## Architecture notes

- Dual-path encoder processes time and frequency axes alternately (reshape trick in `DualPathEncoder.forward`).
- Band-split projection divides frequency into 32 overlapping bands before the transformer; band-merge reconstructs with tapered overlap windows.
- Audio-conditioned FiLM modulation: raw waveform is projected via Conv1d to produce per-band scale/shift for the spectral features.
- Output is 4 complex masks (vocal L/R, instrumental L/R) applied to mixture STFT, then iSTFT'd.
- Model uses custom PoPE (Phase-aware Position Encoding) — not standard rotary embeddings.

## Gotchas

- `Dataset.__len__` returns hardcoded 50000 (infinite-style sampling), not actual dataset size.
- `Dataset.__getitem__` silently retries on any exception — corrupt audio files won't crash training but will slow it.
- `tools/` scripts import `from mvsep import NeuralModel` — they must be run from `main/` or have `main/` on `sys.path`.
- `inference()` in `mvsep.py` and `tools/batch_process.py` are separate implementations with slightly different defaults (chunk_size, overlap). Each `tools/` script also has its own custom inference class — if you change the inference logic in `mvsep.py`, you must make equivalent changes in both `batch_process.py` and `multisong_eval.py`.
- No `__init__.py` in `tools/` — not a package, just standalone scripts.
