import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import math
import warnings
from mvsep import NeuralModel, _load_model_state_dict

warnings.filterwarnings("ignore")

def inference(model, checkpoint_path, input_dir, output_dir, chunk_size=352800, overlap=88200, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_model_state_dict(model, checkpoint_data['model_state_dict'], context="infer")
    model.eval().to(device)

    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.flac'))]
    input_files.sort()

    for filename in input_files:
        input_path = os.path.join(input_dir, filename)
        wav_name = os.path.splitext(filename)[0]
        song_output_dir = os.path.join(output_dir, wav_name)
        os.makedirs(song_output_dir, exist_ok=True)

        output_instrumental_path = os.path.join(song_output_dir, 'instrumental.flac')
        output_vocal_path = os.path.join(song_output_dir, 'vocals.flac')

        input_audio, sr = torchaudio.load(input_path)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
            input_audio = resampler(input_audio)
            sr = 44100
        if input_audio.shape[0] == 1:
            input_audio = input_audio.repeat(2, 1)
        elif input_audio.shape[0] != 2:
            print(f"Skipping {filename}: unsupported channels {input_audio.shape[0]}")
            continue
        input_audio = input_audio.to(device)

        total_length = input_audio.shape[1]
        vocals = torch.zeros_like(input_audio)
        instrumentals = torch.zeros_like(input_audio)
        sum_fade_windows = torch.zeros(total_length, device=device)

        n_fft, hop_length = 4096, 1024
        window = torch.hann_window(n_fft).to(device)
        step_size = chunk_size - overlap

        with tqdm(total=math.ceil(total_length / step_size), desc=f"Processing {filename}", leave=False) as pbar:
            for start in range(0, total_length, step_size):
                end = min(start + chunk_size, total_length)
                chunk, L = input_audio[:, start:end], end - start

                if L < n_fft:
                    pbar.update(1)
                    continue

                target_length = chunk_size
                padded_chunk = chunk
                if L < target_length:
                    padding_amount = target_length - L
                    padded_chunk = F.pad(chunk, (0, padding_amount), 'constant', 0)

                spec = torch.stft(padded_chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True)

                with torch.no_grad():
                    pred_output = model(spec.unsqueeze(0), padded_chunk.unsqueeze(0)).squeeze(0)

                _, F_spec, T_spec = spec.shape
                pred_output_reshaped = pred_output.view(2, 4, F_spec, T_spec)
                pred_real, pred_imag = pred_output_reshaped[0], pred_output_reshaped[1]

                pred_masks_real = pred_real[:4]
                pred_masks_imag = pred_imag[:4]

                vL_cmask = pred_masks_real[0] + 1j * pred_masks_imag[0]
                vR_cmask = pred_masks_real[1] + 1j * pred_masks_imag[1]
                iL_cmask = pred_masks_real[2] + 1j * pred_masks_imag[2]
                iR_cmask = pred_masks_real[3] + 1j * pred_masks_imag[3]

                instrumental_spec = torch.stack([iL_cmask * spec[0], iR_cmask * spec[1]], dim=0)
                vocal_spec = torch.stack([vL_cmask * spec[0], vR_cmask * spec[1]], dim=0)

                vocal_chunk_padded = torch.istft(vocal_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=target_length, center=True)
                inst_chunk_padded = torch.istft(instrumental_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=target_length, center=True)

                vocal_chunk = vocal_chunk_padded[:, :L]
                inst_chunk = inst_chunk_padded[:, :L]

                fade_window = torch.ones(L, device=device)
                effective_overlap = min(L, overlap)

                if start > 0:
                    fade_window[:effective_overlap] = torch.linspace(0, 1, effective_overlap, device=device)
                
                if start + L < total_length:
                    fade_window[-effective_overlap:] = torch.linspace(1, 0, effective_overlap, device=device)

                vocals[:, start:end] += vocal_chunk * fade_window
                instrumentals[:, start:end] += inst_chunk * fade_window
                sum_fade_windows[start:end] += fade_window
                pbar.update(1)

        divisor = sum_fade_windows
        divisor = divisor.clamp(min=1e-8)

        vocals = vocals / divisor
        instrumentals = instrumentals / divisor

        torchaudio.save(output_vocal_path, vocals.cpu().clamp(-1.0, 1.0), sr, format='flac')
        torchaudio.save(output_instrumental_path, instrumentals.cpu().clamp(-1.0, 1.0), sr, format='flac')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralModel()

    if args.infer:
        inference(model, args.checkpoint_path, args.input_dir, args.output_dir, device=device)

if __name__ == '__main__':
    main()
