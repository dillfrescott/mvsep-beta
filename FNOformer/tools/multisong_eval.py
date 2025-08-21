import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import math
from einops import rearrange
from neuralop.models import FNO
import warnings

warnings.filterwarnings("ignore")

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

class TalkingHeadsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.pre_attn_proj = nn.Conv2d(heads, heads, 1, bias=False)
        self.post_attn_proj = nn.Conv2d(heads, heads, 1, bias=False)

    def forward(self, x, rotary_pos_emb=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv
        )
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb(rotary_pos_emb, q)
            k = apply_rotary_pos_emb(rotary_pos_emb, k)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = self.pre_attn_proj(dots)
        attn = dots.softmax(dim=-1)
        attn = self.post_attn_proj(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, fno_n_modes=(16,), fno_hidden_channels=128):
        super().__init__()
        self.attn = TalkingHeadsAttention(dim, heads=heads, dim_head=dim_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fno = FNO(
            n_modes=fno_n_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=dim,
            out_channels=dim
        )

    def forward(self, x, rotary_pos_emb=None):
        x = self.norm1(x)
        attn_out = self.attn(x, rotary_pos_emb=rotary_pos_emb)
        x = x + attn_out
        x = self.norm2(x)
        fno_in = rearrange(x, "b t d -> b d t")
        fno_out = self.fno(fno_in)
        fno_out = rearrange(fno_out, "b d t -> b t d")
        fno_out = F.gelu(fno_out)
        x = x + fno_out
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_head=64, fno_n_modes=(16,), fno_hidden_channels=128):
        super().__init__()
        self.rotary_embedding = RotaryEmbedding(dim_head)
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim,
                heads=heads,
                dim_head=dim_head,
                fno_n_modes=fno_n_modes,
                fno_hidden_channels=fno_hidden_channels
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        b, n, _ = x.shape
        rotary_pos_emb = self.rotary_embedding(n, x.device)
        for layer in self.layers:
            x = layer(x, rotary_pos_emb=rotary_pos_emb)
        return x

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049, embed_dim=512):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim
        self.input_proj_stft = nn.Linear(freq_bins * in_channels, embed_dim)
        
        self.model = Transformer(
            dim=embed_dim,
            depth=6,
            heads=8,
            dim_head=64,
            fno_n_modes=(16,),
            fno_hidden_channels=128
        )
        
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks * 2)

    def forward(self, x_stft_mag, x_audio):
        B, C, F, T = x_stft_mag.shape
        x_stft_mag = x_stft_mag.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x = self.input_proj_stft(x_stft_mag)
        x = self.model(x)
        x = torch.tanh(x)
        x = self.output_proj(x)
        current_T = x.shape[1]
        x = x.view(B, current_T, self.out_masks * 2, F).permute(0, 2, 3, 1)
        return x

def inference(model, checkpoint_path, input_dir, output_dir, chunk_size=485100, overlap=88200, device='cpu'):
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
                mag = torch.abs(spec)

                with torch.no_grad():
                    pred_output = model(mag.unsqueeze(0), chunk.unsqueeze(0)).squeeze(0)

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
        inference(model, args.checkpoint_path, args.input_dir,
                  args.output_dir, device=device)

if __name__ == '__main__':
    main()