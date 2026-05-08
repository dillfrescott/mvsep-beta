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

def inference(model, checkpoint_path, input_dir, output_dir, chunk_size=485100, overlap=88200, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval().to(device)

    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.flac'))]
    input_files.sort()

    n_fft, hop_length = 4096, 1024
    window = torch.hann_window(n_fft).to(device)

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
            continue
        input_audio = input_audio.to(device)

        orig_len = input_audio.shape[1]
        pad_len = (orig_len // hop_length + 1) * hop_length
        input_audio = F.pad(input_audio, (0, pad_len - orig_len))

        with torch.no_grad():
            spec = torch.stft(input_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True)
            x_s = model.band_split(spec.unsqueeze(0))
            x_a = model.audio_proj(input_audio.unsqueeze(0))
            
            if x_a.shape[-1] != x_s.shape[2]:
                x_a = F.interpolate(x_a, size=x_s.shape[2], mode="linear", align_corners=False)

            x_a_t = x_a.permute(0, 2, 1)
            film = model.audio_to_film(x_a_t)
            film = film.view(x_a_t.shape[0], x_a_t.shape[1], model.n_bands, 2, model.embed_dim)
            scale = (1 + 0.1 * torch.tanh(film[:, :, :, 0])).permute(0, 2, 1, 3)
            shift = film[:, :, :, 1].permute(0, 2, 1, 3)

            x = scale * x_s + shift
            x = model.norm(x)

            B, BANDS, T_total, E = x.shape
            chunk_frames = (chunk_size // hop_length)
            overlap_frames = (overlap // hop_length)
            step_frames = chunk_frames - overlap_frames
            
            x_out = torch.zeros_like(x)
            w_sum = torch.zeros(1, 1, T_total, 1, device=device)
            ola_w = torch.hann_window(chunk_frames, device=device).view(1, 1, -1, 1)

            for i in tqdm(range(0, T_total, step_frames), desc=f"Processing {filename}", leave=False):
                start = i
                end = min(i + chunk_frames, T_total)
                L = end - start
                
                if L < chunk_frames:
                    curr_w = torch.hann_window(L, device=device).view(1, 1, -1, 1)
                else:
                    curr_w = ola_w
                    
                x_chunk = x[:, :, start:end, :]
                if L < chunk_frames:
                    x_chunk = F.pad(x_chunk, (0, 0, 0, chunk_frames - L))
                    
                x_res = model.model(x_chunk)
                x_out[:, :, start:end, :] += x_res[:, :, :L, :] * curr_w
                w_sum[:, :, start:end, :] += curr_w

            x = x_out / w_sum.clamp(min=1e-8)
            x = model.band_merge(x)
            x = x + model.merge_refine(x)
            
            B_f, _, F_spec, T_spec = x.shape
            x_reshaped = x.view(B_f, 2, 4, F_spec, T_spec)
            
            vL_c = x_reshaped[0, 0, 0] + 1j * x_reshaped[0, 1, 0]
            vR_c = x_reshaped[0, 0, 1] + 1j * x_reshaped[0, 1, 1]
            iL_c = x_reshaped[0, 0, 2] + 1j * x_reshaped[0, 1, 2]
            iR_c = x_reshaped[0, 0, 3] + 1j * x_reshaped[0, 1, 3]

            v_spec = torch.stack([vL_c * spec[0], vR_c * spec[1]], dim=0)
            i_spec = torch.stack([iL_c * spec[0], iR_c * spec[1]], dim=0)

            vocals = torch.istft(v_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=orig_len, center=True)
            instrs = torch.istft(i_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=orig_len, center=True)

        torchaudio.save(output_vocal_path, vocals.cpu().clamp(-1.0, 1.0), sr, format='flac')
        torchaudio.save(output_instrumental_path, instrs.cpu().clamp(-1.0, 1.0), sr, format='flac')

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
