import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast
from mvsep import NeuralModel, clean_state_dict

warnings.filterwarnings("ignore")

def inference(model, checkpoint_data, input_dir, output_dir, chunk_size=264600, overlap=88200, device='cpu'):
    stems = checkpoint_data.get('stems', ['vocals', 'other'])
    num_stems = len(stems)
    if 'ema_state_dict' in checkpoint_data:
        print("Loading EMA weights for inference.")
        model.load_state_dict(clean_state_dict(checkpoint_data['ema_state_dict']), strict=False)
    else:
        print("Loading regular model weights for inference.")
        model.load_state_dict(clean_state_dict(checkpoint_data['model_state_dict']), strict=False)
    model.eval().to(device)

    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.flac'))]
    input_files.sort()

    for filename in input_files:
        input_path = os.path.join(input_dir, filename)
        wav_name = os.path.splitext(filename)[0]
        song_output_dir = os.path.join(output_dir, wav_name)
        os.makedirs(song_output_dir, exist_ok=True)

        input_audio, sr = torchaudio.load(input_path)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
            input_audio = resampler(input_audio)
            sr = 44100
        if input_audio.shape[0] == 1:
            input_audio = input_audio.repeat(2, 1)
        elif input_audio.shape[0] != 2:
            continue
        input_audio = input_audio.to(device)

        total_length = input_audio.shape[1]
        pred_stems = [torch.zeros_like(input_audio) for _ in range(num_stems)]
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
                if L < target_length:
                    pad_amount = target_length - L
                    max_reflect = L - 1
                    if pad_amount <= max_reflect:
                        padded_chunk = F.pad(chunk, (0, pad_amount), mode='reflect')
                    else:
                        padded_chunk = F.pad(chunk, (0, max_reflect), mode='reflect')
                        padded_chunk = F.pad(padded_chunk, (0, pad_amount - max_reflect), mode='constant')

                spec = torch.stft(padded_chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True)

                device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
                if device_type == 'cuda' and torch.cuda.is_bf16_supported():
                    autocast_ctx = autocast(dtype=torch.bfloat16)
                else:
                    autocast_ctx = autocast()

                with torch.no_grad():
                    with autocast_ctx:
                        pred_output = model(spec.unsqueeze(0)).squeeze(0)

                _, F_spec, T_spec = spec.shape
                pred_output_reshaped = pred_output.contiguous().view(2, num_stems * 2, F_spec, T_spec)
                pred_real, pred_imag = pred_output_reshaped[0], pred_output_reshaped[1]

                if L == chunk_size:
                    w = ola_window
                else:
                    w = torch.hann_window(L, device=device, dtype=input_audio.dtype)

                if num_chunks > 1:
                    if chunk_idx == 0:
                        w = w.clone()
                        w[:L // 2] = 1.0
                    elif end >= total_length:
                        w = w.clone()
                        w[L // 2:] = 1.0
                else:
                    w = torch.ones(L, device=device, dtype=input_audio.dtype)

                actual_end = min(start + L, total_length)
                usable = actual_end - start

                for j in range(num_stems):
                    cmask = pred_real[2*j:2*j+2] + 1j * pred_imag[2*j:2*j+2]
                    stem_spec = cmask * spec
                    stem_chunk = torch.istft(stem_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=target_length, center=True)
                    pred_stems[j][:, start:actual_end] += stem_chunk[:, :usable] * w[:usable]

                sum_weights[start:actual_end] += w[:usable]
                pbar.update(1)

        sum_weights = sum_weights.clamp(min=1e-8)
        reconstructed_stems = [pred_stems[j] / sum_weights for j in range(num_stems)]
        residual = input_audio - sum(reconstructed_stems)
        for j in range(num_stems):
            res = (reconstructed_stems[j] + (1.0 / num_stems) * residual).clamp(-1.0, 1.0)
            torchaudio.save(os.path.join(song_output_dir, f'{stems[j]}.flac'), res.cpu(), sr, format='flac')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_data = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    stems = checkpoint_data.get('stems', ['vocals', 'other'])
    model = NeuralModel(sources=len(stems))

    if args.infer:
        inference(model, checkpoint_data, args.input_dir, args.output_dir, device=device)

if __name__ == '__main__':
    main()
