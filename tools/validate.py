import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import math
from conformer import Conformer

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049, embed_dim=512):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim
        self.input_proj_stft = nn.Linear(freq_bins * in_channels, embed_dim)
        self.model = Conformer(
            dim=embed_dim,
            depth=8,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.1,
            ff_dropout=0.1,
            conv_dropout=0.1
        )
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks * 2)

    def forward(self, x_stft_mag, x_audio):
        B, C, F, T = x_stft_mag.shape
        x_stft_mag = x_stft_mag.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x = self.input_proj_stft(x_stft_mag)
        x = self.model(x)
        x = self.output_proj(x)
        current_T = x.shape[1]
        x = x.view(B, current_T, self.out_masks * 2, F).permute(0, 2, 3, 1)
        return x

def compute_sdr(ref, est, eps=1e-8):
    num = (ref ** 2).sum(dim=-1)
    den = ((ref - est) ** 2).sum(dim=-1) + eps
    sdr = 10.0 * torch.log10((num + eps) / den)
    return sdr.mean().item()

def find_stem_pair(track_dir):
    candidates = [
        (os.path.join(track_dir, "vocals.wav"), os.path.join(track_dir, "other.wav")),
        (os.path.join(track_dir, "vocals.flac"), os.path.join(track_dir, "other.flac")),
        (os.path.join(track_dir, "vocals.wav"), os.path.join(track_dir, "other.flac")),
        (os.path.join(track_dir, "vocals.flac"), os.path.join(track_dir, "other.wav")),
    ]
    for v, o in candidates:
        if os.path.isfile(v) and os.path.isfile(o):
            return v, o
    return None, None

def validate(model, checkpoint_path, test_dir, chunk_size=352800, overlap=88200, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval().to(device)

    track_dirs = [d for d in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, d))]

    vocal_sdrs = []
    other_sdrs = []

    with tqdm(total=len(track_dirs), desc="Validating tracks") as main_pbar:
        for track in track_dirs:
            track_dir = os.path.join(test_dir, track)
            vocal_ref_path, other_ref_path = find_stem_pair(track_dir)
            if not (vocal_ref_path and other_ref_path):
                main_pbar.update(1)
                continue

            vocal_ref, sr_v = torchaudio.load(vocal_ref_path)
            other_ref, sr_o = torchaudio.load(other_ref_path)

            target_sr = 44100
            if sr_v != target_sr:
                vocal_ref = torchaudio.transforms.Resample(orig_freq=sr_v, new_freq=target_sr)(vocal_ref)
            if sr_o != target_sr:
                other_ref = torchaudio.transforms.Resample(orig_freq=sr_o, new_freq=target_sr)(other_ref)

            if vocal_ref.shape[0] == 1:
                vocal_ref = vocal_ref.repeat(2, 1)
            if other_ref.shape[0] == 1:
                other_ref = other_ref.repeat(2, 1)

            min_len = min(vocal_ref.shape[1], other_ref.shape[1])
            vocal_ref = vocal_ref[:, :min_len]
            other_ref = other_ref[:, :min_len]
            mix_audio = (vocal_ref + other_ref).clamp(-1.0, 1.0)

            mix_audio = mix_audio.to(device)
            vocal_ref = vocal_ref.to(device)
            other_ref = other_ref.to(device)

            total_length = mix_audio.shape[1]
            vocals = torch.zeros_like(mix_audio)
            others = torch.zeros_like(mix_audio)

            n_fft = 4096
            hop_length = 1024
            window = torch.hann_window(n_fft).to(device)
            min_chunk = n_fft

            step_size = max(1, chunk_size - overlap)
            cross_fade = overlap // 2
            num_chunks = math.ceil(max(0, total_length - overlap) / step_size)

            with torch.no_grad():
                for i in range(0, total_length, step_size):
                    end = min(i + chunk_size, total_length)
                    chunk = mix_audio[:, i:end]
                    L = chunk.shape[1]

                    if L < min_chunk:
                        if i == 0:
                            pad_amt = min_chunk - L
                            chunk = F.pad(chunk, (0, pad_amt))
                            L = chunk.shape[1]
                        else:
                            continue

                    spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True)
                    mag = torch.abs(spec)

                    pred_output = model(mag.unsqueeze(0), chunk.unsqueeze(0))
                    pred_output = pred_output.squeeze(0)

                    _, F_spec, T_spec = pred_output.shape
                    pred_masks = pred_output.view(2, 4, F_spec, T_spec)
                    pred_masks_real = pred_masks[0]
                    pred_masks_imag = pred_masks[1]

                    vL_cmask = pred_masks_real[0] + 1j * pred_masks_imag[0]
                    vR_cmask = pred_masks_real[1] + 1j * pred_masks_imag[1]
                    oL_cmask = pred_masks_real[2] + 1j * pred_masks_imag[2]
                    oR_cmask = pred_masks_real[3] + 1j * pred_masks_imag[3]

                    other_spec = torch.stack([oL_cmask * spec[0], oR_cmask * spec[1]], dim=0)
                    vocal_spec = torch.stack([vL_cmask * spec[0], vR_cmask * spec[1]], dim=0)

                    vocal_chunk = torch.istft(vocal_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=L, center=True)
                    other_chunk = torch.istft(other_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=L, center=True)

                    if i == 0:
                        vocals[:, :L] = vocal_chunk
                        others[:, :L] = other_chunk
                    else:
                        fade_in = torch.linspace(0, 1, cross_fade, device=device)
                        fade_out = 1 - fade_in
                        ov_end = min(i + cross_fade, total_length)
                        actual = ov_end - i

                        vocals[:, i:ov_end] = vocals[:, i:ov_end] * fade_out[:actual] + vocal_chunk[:, :actual] * fade_in[:actual]
                        others[:, i:ov_end] = others[:, i:ov_end] * fade_out[:actual] + other_chunk[:, :actual] * fade_in[:actual]

                        tail_start = i + cross_fade
                        tail_end = min(i + L, total_length)
                        if tail_start < tail_end:
                            vocals[:, tail_start:tail_end] = vocal_chunk[:, tail_start - i:tail_end - i]
                            others[:, tail_start:tail_end] = other_chunk[:, tail_start - i:tail_end - i]

            vocals = vocals[:, :total_length].clamp(-1.0, 1.0)
            others = others[:, :total_length].clamp(-1.0, 1.0)

            v_sdr = compute_sdr(vocal_ref, vocals)
            o_sdr = compute_sdr(other_ref, others)
            vocal_sdrs.append(v_sdr)
            other_sdrs.append(o_sdr)

            main_pbar.update(1)

    avg_vocal_sdr = sum(vocal_sdrs) / len(vocal_sdrs) if len(vocal_sdrs) > 0 else float('nan')
    avg_other_sdr = sum(other_sdrs) / len(other_sdrs) if len(other_sdrs) > 0 else float('nan')
    avg_total_sdr = (avg_vocal_sdr + avg_other_sdr) / 2.0 if (not math.isnan(avg_vocal_sdr) and not math.isnan(avg_other_sdr)) else float('nan')

    print(f"Average Vocal SDR: {avg_vocal_sdr:.4f} dB")
    print(f"Average Other SDR: {avg_other_sdr:.4f} dB")
    print(f"Average Total SDR: {avg_total_sdr:.4f} dB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralModel()
    validate(model, args.checkpoint_path, args.test_dir, device=device)

if __name__ == '__main__':
    main()
