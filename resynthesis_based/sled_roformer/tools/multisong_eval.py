import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import math
import warnings

warnings.filterwarnings("ignore")

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

class SledAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos_emb):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads, -1).transpose(1, 2), qkv)

        q = apply_rotary_pos_emb(pos_emb, q)
        k = apply_rotary_pos_emb(pos_emb, k)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SledTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SledAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, pos_emb):
        x = self.attn(self.norm1(x), pos_emb=pos_emb) + x
        x = self.ff(self.norm2(x)) + x
        return x

class SledEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(SledTransformerBlock(dim, heads, dim_head, mlp_dim, dropout))
        self.rotary_emb = RotaryEmbedding(dim_head)

    def forward(self, x):
        B, N, C = x.shape
        pos_emb = self.rotary_emb(N, x.device)
        
        layer_outputs = []
        current_input = x
        for layer in self.layers:
            current_input = layer(current_input, pos_emb=pos_emb)
            layer_outputs.append(current_input)
            
        return layer_outputs

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049,
                 embed_dim=1024, depth=8, heads=16):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.num_streams = sources * in_channels
        self.embed_dim = embed_dim
        self.input_proj_stft = nn.Linear(freq_bins * in_channels * 2, embed_dim)
        
        self.model = SledEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=embed_dim * 4,
            dim_head=embed_dim // heads
        )
        self.sled_weights = nn.Parameter(torch.ones(depth))
        
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.num_streams * 2)

    def forward(self, x_stft, x_audio):
        x_stft = torch.cat([x_stft.real, x_stft.imag], dim=1)
        B, C, F_dim, T = x_stft.shape
        x_stft = x_stft.permute(0, 3, 1, 2).contiguous().view(B, T, C * F_dim)
        x = self.input_proj_stft(x_stft)
        
        layer_outputs = self.model(x)
        
        projections = torch.stack([self.output_proj(torch.tanh(out)) for out in layer_outputs])
        
        sled_weights_softmax = F.softmax(self.sled_weights, dim=0).view(-1, 1, 1, 1)
        x = torch.sum(projections * sled_weights_softmax, dim=0)
        
        current_T = x.shape[1]
        x = x.view(B, current_T, self.num_streams * 2, F_dim).permute(0, 2, 3, 1)
        return x

def _reshape_pred(output, F, T):
    if output.dim() == 4:
        B = output.shape[0]
        return output.view(B, 2, 4, F, T)
    elif output.dim() == 3:
        return output.view(2, 4, F, T)
    else:
        raise ValueError(f"Unexpected prediction shape: {output.shape}")

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

                with torch.no_grad():
                    pred_output = model(spec.unsqueeze(0), chunk.unsqueeze(0)).squeeze(0)

                _, F_spec, T_spec = spec.shape
                pred_output_reshaped = _reshape_pred(pred_output, F_spec, T_spec)
                pred_mag_raw, pred_phase_raw = pred_output_reshaped[0], pred_output_reshaped[1]

                pred_mag_log1p = F.gelu(pred_mag_raw)
                
                pred_vocal_phase_raw_split = pred_phase_raw[:2]
                pred_instr_phase_raw_split = pred_phase_raw[2:]
                pred_vocal_phase = torch.tanh(pred_vocal_phase_raw_split) * math.pi
                pred_instr_phase = torch.tanh(pred_instr_phase_raw_split) * math.pi

                pred_vocal_mag_log1p = pred_mag_log1p[:2]
                pred_instr_mag_log1p = pred_mag_log1p[2:]

                pred_vocal_mag_linear = torch.expm1(pred_vocal_mag_log1p).clamp(min=0.0)
                pred_instr_mag_linear = torch.expm1(pred_instr_mag_log1p).clamp(min=0.0)

                vocal_spec = pred_vocal_mag_linear * torch.exp(1j * pred_vocal_phase)
                instrumental_spec = pred_instr_mag_linear * torch.exp(1j * pred_instr_phase)

                vocal_chunk = torch.istft(vocal_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=L, center=True)
                inst_chunk = torch.istft(instrumental_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=L, center=True)

                fade_window = torch.ones(L, device=device)
                effective_overlap = min(L, overlap)

                if start > 0:
                    fade_in = torch.hann_window(effective_overlap * 2, periodic=False, device=device)[:effective_overlap]
                    fade_window[:effective_overlap] = fade_in
                
                if start + L < total_length:
                    fade_out = torch.hann_window(effective_overlap * 2, periodic=False, device=device)[effective_overlap:]
                    fade_window[-effective_overlap:] = fade_out

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