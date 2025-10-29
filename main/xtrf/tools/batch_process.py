import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import math
from x_transformers import Encoder
import warnings

warnings.filterwarnings("ignore")

class FreqMixerBlock(nn.Module):
    def __init__(self, channels, k=5, dilation=1, p_drop=0.1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv2d(channels, channels, (k, 1), padding=(pad, 0), dilation=(dilation, 1), groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        r = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return r + self.res_scale * x

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049, embed_dim=512, depth=8, heads=8,
                 freq_groups=8, drop_nyquist=True, mixer_blocks=2, mixer_kernel=7,
                 mixer_dilations=(1, 2), mixer_dropout=0.1):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim
        self.freq_groups = freq_groups
        self.drop_nyquist = drop_nyquist

        grouped_freq_bins = freq_bins - 1 if drop_nyquist else freq_bins
        self.grouped_freq_bins = grouped_freq_bins
        self.group_size = math.ceil(grouped_freq_bins / freq_groups)
        self.grouped_target_bins = self.group_size * freq_groups
        self.freq_pad = self.grouped_target_bins - grouped_freq_bins

        self.input_proj_stft = nn.Linear(self.group_size * in_channels * 2, embed_dim)
        self.model = Encoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=True
        )
        self.output_proj = nn.Linear(embed_dim, self.group_size * self.out_masks * 2, bias=False)

        layers = []
        c = self.out_masks * 2
        if not mixer_dilations:
            mixer_dilations = (1,) * mixer_blocks
        for i in range(mixer_blocks):
            d = mixer_dilations[i] if i < len(mixer_dilations) else 1
            layers.append(FreqMixerBlock(c, k=mixer_kernel, dilation=d, p_drop=mixer_dropout))
        self.freq_mixer = nn.Sequential(*layers)

        self.final_activation = nn.Tanh()

    def forward(self, x_stft, x_audio=None):
        x_stft = torch.cat([x_stft.real, x_stft.imag], dim=1)
        x_mag = x_stft[:, :, :self.grouped_freq_bins, :] if self.drop_nyquist else x_stft
        B, C2, Fg, T = x_mag.shape

        if self.freq_pad > 0:
            x_mag = F.pad(x_mag, (0, 0, 0, self.freq_pad))
            Fg_p = Fg + self.freq_pad
        else:
            Fg_p = Fg

        x_mag = x_mag.view(B, C2, self.freq_groups, self.group_size, T)
        x_tok = x_mag.permute(0, 2, 4, 1, 3).contiguous().view(B * self.freq_groups, T, C2 * self.group_size)

        x = self.input_proj_stft(x_tok)
        x = self.model(x)
        x = self.output_proj(x)

        x = x.view(B, self.freq_groups, T, self.out_masks * 2, self.group_size)
        x = x.permute(0, 3, 1, 4, 2).contiguous().view(B, self.out_masks * 2, Fg_p, T)

        if self.freq_pad > 0:
            x = x[:, :, :self.grouped_freq_bins, :]

        x = self.freq_mixer(x)
        x = self.final_activation(x)

        if self.drop_nyquist:
            nyq = x[:, :, -1:, :].clone()
            x = torch.cat([x, nyq], dim=2)

        return x

def inference(model, checkpoint_path, input_dir, output_dir, chunk_size=485100, overlap=88200, device='cpu'):
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

                spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True)

                with torch.no_grad():
                    pred_output = model(spec.unsqueeze(0), chunk.unsqueeze(0)).squeeze(0)

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

                vocal_chunk = torch.istft(vocal_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=L, center=True)
                inst_chunk = torch.istft(instrumental_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=L, center=True)

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