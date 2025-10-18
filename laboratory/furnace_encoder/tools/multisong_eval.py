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

class FurnaceEncoder(nn.Module):
    def __init__(self, dim, depth=12, heads=8, rotary_pos_emb=True, max_mem_len=8192, decay=0.0):
        super().__init__()
        self.core = Encoder(dim=dim, depth=depth, heads=heads, rotary_pos_emb=rotary_pos_emb, cross_attend=True)
        self.max_mem_len = max_mem_len
        self.register_buffer('_memory', None, persistent=False)
        self.decay = float(decay)

    def reset_state(self):
        self._memory = None

    def _ensure_memory_for_batch(self, x):
        B = x.size(0)
        if self._memory is None:
            return None
        if self._memory.size(0) != B:
            self._memory = None
            return None
        return self._memory

    def forward(self, x, reset_state=False):
        if reset_state:
            self.reset_state()
            
        B, T_in, D = x.shape
        mem = self._ensure_memory_for_batch(x)

        if mem is None or mem.size(1) == 0:
            dummy_context = torch.zeros(B, 1, D, device=x.device)
            out = self.core(x, context=dummy_context)
            new_mem = out[:, -self.max_mem_len:, :].detach()
            self._memory = new_mem
            return out

        out = self.core(x, context=mem)

        new_mem = out[:, -self.max_mem_len:, :].detach()
        if self.decay and self._memory is not None:
            prev = self._memory
            if prev.size(1) < new_mem.size(1):
                pad = new_mem.size(1) - prev.size(1)
                prev = torch.cat([x.new_zeros(B, pad, D), prev], dim=1)
            elif prev.size(1) > new_mem.size(1):
                prev = prev[:, -new_mem.size(1):, :]
            blended = self.decay * prev + (1.0 - self.decay) * new_mem
            self._memory = blended.detach()
        else:
            self._memory = new_mem
        return out

    def think(self, steps=1):
        if self._memory is None or self._memory.size(1) == 0:
            return
        for _ in range(int(steps)):
            mem = self._memory
            out = self.core(mem, context=mem)
            new_mem = out[:, -self.max_mem_len:, :].detach()
            if self.decay:
                prev = mem
                if prev.size(1) < new_mem.size(1):
                    pad = new_mem.size(1) - prev.size(1)
                    prev = torch.cat([mem.new_zeros(mem.size(0), pad, mem.size(2)), prev], dim=1)
                elif prev.size(1) > new_mem.size(1):
                    prev = prev[:, -new_mem.size(1):, :]
                self._memory = (self.decay * prev + (1.0 - self.decay) * new_mem).detach()
            else:
                self._memory = new_mem

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049,
                 embed_dim=512, depth=12, heads=8,
                 max_mem_len=8192, memory_decay=0.0):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim
        self.input_proj_stft = nn.Linear(freq_bins * in_channels * 2, embed_dim)
        self.model = FurnaceEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=True,
            max_mem_len=max_mem_len,
            decay=memory_decay
        )
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks * 2)

    def reset_memory(self):
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state()

    def think(self, steps=1):
        if hasattr(self.model, 'think'):
            self.model.think(steps)

    def forward(self, x_stft, x_audio, reset_state=False):
        x_stft = torch.cat([x_stft.real, x_stft.imag], dim=1)
        B, C, F, T = x_stft.shape
        x_stft = x_stft.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x = self.input_proj_stft(x_stft)
        x = self.model(x, reset_state=reset_state)
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
        inference(model, args.checkpoint_path, args.input_dir,
                  args.output_dir, device=device)

if __name__ == '__main__':
    main()