import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import math
import warnings
from mvsep import NeuralModel

warnings.filterwarnings("ignore")

def inference(model, checkpoint_path, input_dir, output_dir, chunk_size=264600, overlap=88200, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
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
        sum_weights = torch.zeros(total_length, device=device)

        n_fft, hop_length = 4096, 1024
        window = torch.hann_window(n_fft).to(device)
        step_size = chunk_size - overlap

        chunk_starts = list(range(0, total_length, step_size))
        num_chunks = len(chunk_starts)

        ola_window = torch.hann_window(chunk_size, device=device, dtype=input_audio.dtype)

        with tqdm(total=num_chunks, desc=f"Processing {filename}", leave=False) as pbar:
            for chunk_idx, i in enumerate(chunk_starts):
                start = i
                end = min(i + chunk_size, total_length)
                chunk = input_audio[:, start:end]
                L = chunk.shape[1]

                if L < n_fft:
                    needed = n_fft - L
                    left_ctx = min(needed, start)
                    right_ctx = min(needed - left_ctx, total_length - end)
                    extended_start = start - left_ctx
                    extended_end = end + right_ctx
                    chunk = input_audio[:, extended_start:extended_end]
                    start = extended_start
                    end = extended_end
                    L = chunk.shape[1]
                    if L < n_fft:
                        chunk = F.pad(chunk, (0, n_fft - L), mode='reflect' if L > 1 else 'constant')
                        L = chunk.shape[1]

                target_length = chunk_size
                padded_chunk = chunk
                pad_amount = 0
                if L < target_length:
                    pad_amount = target_length - L
                    max_reflect = L - 1
                    if pad_amount <= max_reflect:
                        padded_chunk = F.pad(chunk, (0, pad_amount), mode='reflect')
                    else:
                        padded_chunk = F.pad(chunk, (0, max_reflect), mode='reflect')
                        remaining = pad_amount - max_reflect
                        padded_chunk = F.pad(padded_chunk, (0, remaining), mode='constant')

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

                vocal_chunk_full = torch.istft(vocal_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=target_length, center=True)
                inst_chunk_full = torch.istft(instrumental_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=target_length, center=True)

                vocal_chunk = vocal_chunk_full[:, :L]
                inst_chunk = inst_chunk_full[:, :L]

                if L == chunk_size:
                    w = ola_window
                else:
                    w = torch.hann_window(L, device=device, dtype=input_audio.dtype)

                if num_chunks == 1:
                    w = torch.ones(L, device=device, dtype=input_audio.dtype)
                else:
                    if chunk_idx == 0:
                        w = w.clone()
                        half = L // 2
                        w[:half] = 1.0
                    elif end >= total_length:
                        w = w.clone()
                        half = L // 2
                        w[half:] = 1.0

                actual_end = min(start + L, total_length)
                usable = actual_end - start

                vocals[:, start:actual_end] += vocal_chunk[:, :usable] * w[:usable]
                instrumentals[:, start:actual_end] += inst_chunk[:, :usable] * w[:usable]
                sum_weights[start:actual_end] += w[:usable]
                pbar.update(1)

        sum_weights = sum_weights.clamp(min=1e-8)

        vocals = vocals / sum_weights
        instrumentals = instrumentals / sum_weights

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
