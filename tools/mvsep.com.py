import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import math
from strassen_attention.strassen_transformer import StrassenTransformer

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049,
                 embed_dim=512, depth=6):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim

        self.input_proj_stft = nn.Linear(freq_bins * in_channels, embed_dim)
        self.model = StrassenTransformer(dim=embed_dim, depth=depth)
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks * 2)

    def forward(self, x_stft_mag, x_audio):
        B, C, F, T = x_stft_mag.shape
        x_stft_mag = x_stft_mag.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x = self.input_proj_stft(x_stft_mag)
        x = self.model(x)
        x = self.output_proj(x)
        current_T = x.shape[1]
        x = x.view(B, current_T, self.out_masks * 2, F).permute(0, 2, 3, 1)
        return torch.sigmoid(x)

def inference(model, checkpoint_path, input_dir, output_dir,
              chunk_size=529200, overlap=88200, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval().to(device)

    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith('_mixture.wav')]
    input_files.sort()

    for filename in input_files:
        input_wav_path = os.path.join(input_dir, filename)
        song_id = filename.replace('_mixture.wav', '')
        output_instrumental_path = os.path.join(output_dir, f'{song_id}_instrum.flac')
        output_vocal_path = os.path.join(output_dir, f'{song_id}_vocals.flac')

        input_audio, sr = torchaudio.load(input_wav_path)
        if sr != 44100:
            continue
        if input_audio.shape[0] == 1:
            input_audio = input_audio.repeat(2, 1)
        elif input_audio.shape[0] != 2:
            continue
        input_audio = input_audio.to(device)

        total_length = input_audio.shape[1]
        vocals = torch.zeros_like(input_audio)
        instrumentals = torch.zeros_like(input_audio)

        n_fft = 4096
        hop_length = 1024
        window = torch.hann_window(n_fft).to(device)
        min_chunk = n_fft

        step_size = max(1, chunk_size - overlap)
        cross_fade = overlap // 2
        num_chunks = math.ceil(max(0, total_length - overlap) / step_size)

        with tqdm(total=num_chunks, desc=f"Processing {filename}") as pbar:
            for i in range(0, total_length, step_size):
                end = min(i + chunk_size, total_length)
                chunk = input_audio[:, i:end]
                L = chunk.shape[1]

                if L < min_chunk:
                    if i == 0:
                        pad_amt = min_chunk - L
                        chunk = F.pad(chunk, (0, pad_amt))
                        L = chunk.shape[1]
                    else:
                        pbar.update(1)
                        continue

                spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length,
                                  window=window, return_complex=True, center=True)
                mag = torch.abs(spec)

                with torch.no_grad():
                    pred_output = model(mag.unsqueeze(0), chunk.unsqueeze(0))
                pred_output = pred_output.squeeze(0)

                _, F_spec, T_spec = pred_output.shape
                pred_masks = pred_output.view(2, 4, F_spec, T_spec)
                pred_masks_real = pred_masks[0]
                pred_masks_imag = pred_masks[1]

                vL_cmask = pred_masks_real[0] + 1j * pred_masks_imag[0]
                vR_cmask = pred_masks_real[1] + 1j * pred_masks_imag[1]
                iL_cmask = pred_masks_real[2] + 1j * pred_masks_imag[2]
                iR_cmask = pred_masks_real[3] + 1j * pred_masks_imag[3]

                instrumental_spec = torch.stack([iL_cmask * spec[0], iR_cmask * spec[1]], dim=0)
                vocal_spec = torch.stack([vL_cmask * spec[0], vR_cmask * spec[1]], dim=0)

                vocal_chunk = torch.istft(vocal_spec, n_fft=n_fft, hop_length=hop_length,
                                          window=window, length=L, center=True)
                inst_chunk = torch.istft(instrumental_spec, n_fft=n_fft, hop_length=hop_length,
                                         window=window, length=L, center=True)

                if i == 0:
                    vocals[:, :L] = vocal_chunk
                    instrumentals[:, :L] = inst_chunk
                else:
                    fade_in = torch.linspace(0, 1, cross_fade, device=device)
                    fade_out = 1 - fade_in
                    ov_end = min(i + cross_fade, total_length)
                    actual = ov_end - i

                    vocals[:, i:ov_end] = vocals[:, i:ov_end] * fade_out[:actual] + vocal_chunk[:, :actual] * fade_in[:actual]
                    instrumentals[:, i:ov_end] = instrumentals[:, i:ov_end] * fade_out[:actual] + inst_chunk[:, :actual] * fade_in[:actual]

                    tail_start = i + cross_fade
                    tail_end = min(i + L, total_length)
                    if tail_start < tail_end:
                        vocals[:, tail_start:tail_end] = vocal_chunk[:, tail_start - i:tail_end - i]
                        instrumentals[:, tail_start:tail_end] = inst_chunk[:, tail_start - i:tail_end - i]

                pbar.update(1)

        vocals = vocals[:, :total_length].clamp(-1.0, 1.0)
        instrumentals = instrumentals[:, :total_length].clamp(-1.0, 1.0)
        torchaudio.save(output_vocal_path, vocals.cpu(), sr, format='flac')
        torchaudio.save(output_instrumental_path, instrumentals.cpu(), sr, format='flac')

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
        inference(model, args.checkpoint_path, args.input_dir,
                  args.output_dir, device=device)

if __name__ == '__main__':
    main()
