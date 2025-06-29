<img src="https://github.com/user-attachments/assets/31d7d5b7-d511-475e-bcae-5f8cf5feb109" width="400" alt="mvsep-beta-logo-2" />

# Music Source Separation with S5 Model

This repository contains an implementation of a music source separation system using the S5 sequence-to-sequence model. The system separates stereo music into vocal and instrumental components.

## Features
- Complex-valued mask prediction using S5 blocks
- Multi-resolution STFT loss for better time-frequency representation
- Efficient training with Prodigy optimizer
- Chunked inference for processing long audio files
- GPU acceleration support

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8 (for GPU training)
- See [requirements.txt](requirements.txt) for full dependencies

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/mvsep.git
cd mvsep
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
For training, organize your dataset with the following structure:
```
train/
  track1/
    vocals.wav
    other.wav
  track2/
    vocals.wav
    other.wav
  ...
```
All files should be stereo WAV format at 44.1kHz sample rate.

## Usage

### Training
```bash
python mvsep.py --train --data_dir path/to/train --batch_size 4 --segment_length 529200 --checkpoint_steps 2000
```
- Training checkpoints will be saved every 2000 steps (adjust with `--checkpoint_steps`)
- To resume training from a checkpoint: add `--checkpoint_path path/to/checkpoint.pt`

### Inference
```bash
python mvsep.py --infer --input_wav input.wav --output_vocal vocals.wav --output_instrumental instrumental.wav --checkpoint_path path/to/checkpoint.pt
```

### Command Line Options
| Option | Description | Default |
|--------|-------------|---------|
| `--train` | Train the model | |
| `--infer` | Run inference | |
| `--data_dir` | Training dataset path | `train` |
| `--batch_size` | Training batch size | `1` |
| `--checkpoint_steps` | Save checkpoint every N steps | `2000` |
| `--checkpoint_path` | Path to checkpoint file | `None` |
| `--input_wav` | Input audio file for inference | `None` |
| `--output_vocal` | Output vocal file path | `output_vocal.wav` |
| `--output_instrumental` | Output instrumental file path | `output_instrumental.wav` |
| `--segment_length` | Training segment length (samples) | `529200` (~12s) |

## Implementation Details
The model uses:
- S5 sequence-to-sequence blocks for temporal modeling
- Complex mask prediction in frequency domain
- Multi-resolution STFT loss function
- Overlap-add inference for long audio files
- Prodigy optimizer with custom complex number support

## Troubleshooting
- **Short audio files**: Ensure all training samples are longer than the segment length
- **NaN loss**: Model automatically skips batches with NaN loss
- **Sample rate**: Input files must be 44.1kHz
- **Memory issues**: Reduce batch size or segment length

## License
This project is licensed under the terms of the WTFPL license. See [LICENSE](LICENSE) for details.
