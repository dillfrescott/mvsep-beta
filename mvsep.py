import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prodigyopt import Prodigy
import random
import math
from typing import Optional, Tuple

def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    b = lengths.shape[0]
    t = int(torch.max(lengths).item())
    return torch.arange(t, device=lengths.device, dtype=lengths.dtype).expand(b, t) >= lengths.unsqueeze(1)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)

def build_rotary_sin_cos(seq_len: int, dim: int, device: torch.device, base: float = 10000.0):
    half = dim // 2
    freq = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freq / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", t, inv_freq)
    sin = torch.sin(freqs)
    cos = torch.cos(freqs)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,T,dim)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,T,dim)
    return sin, cos

class _ConvolutionModule(torch.nn.Module):
    def __init__(self, input_dim: int, num_channels: int, depthwise_kernel_size: int, dropout: float = 0.0, bias: bool = False, use_group_norm: bool = False) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, 2 * num_channels, 1, stride=1, padding=0, bias=bias),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(num_channels, num_channels, depthwise_kernel_size, stride=1, padding=(depthwise_kernel_size - 1) // 2, groups=num_channels, bias=bias),
            (torch.nn.GroupNorm(num_groups=1, num_channels=num_channels) if use_group_norm else torch.nn.BatchNorm1d(num_channels)),
            torch.nn.SiLU(),
            torch.nn.Conv1d(num_channels, input_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)

class _FeedForwardModule(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential(input)

class RotaryMHA(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embedding.")
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T, B, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(T, B, H, Hd).permute(1, 2, 0, 3)  # B,H,T,Hd
        k = self.k_proj(x).view(T, B, H, Hd).permute(1, 2, 0, 3)  # B,H,T,Hd
        v = self.v_proj(x).view(T, B, H, Hd).permute(1, 2, 0, 3)  # B,H,T,Hd

        sin, cos = build_rotary_sin_cos(T, Hd, x.device)
        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k, sin, cos)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)  # B,H,T,T

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # B,1,1,T
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # B,H,T,Hd
        out = out.permute(2, 0, 1, 3).contiguous().view(T, B, D)  # T,B,D
        out = self.out_proj(out)
        return out

class ConformerLayer(torch.nn.Module):
    def __init__(self, input_dim: int, ffn_dim: int, num_attention_heads: int, depthwise_conv_kernel_size: int, dropout: float = 0.0, use_group_norm: bool = False, convolution_first: bool = False) -> None:
        super().__init__()
        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = RotaryMHA(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.conv_module = _ConvolutionModule(input_dim=input_dim, num_channels=input_dim, depthwise_kernel_size=depthwise_conv_kernel_size, dropout=dropout, bias=True, use_group_norm=use_group_norm)
        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        return residual + input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, key_padding_mask)
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        x = self.final_layer_norm(x)
        return x

class Conformer(torch.nn.Module):
    def __init__(self, input_dim: int, num_heads: int, ffn_dim: int, num_layers: int, depthwise_conv_kernel_size: int, dropout: float = 0.0, use_group_norm: bool = False, convolution_first: bool = False):
        super().__init__()
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), lengths

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049,
                 embed_dim=512):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim
        self.input_proj_stft = nn.Linear(freq_bins * in_channels, embed_dim)
        self.model = Conformer(
            input_dim=embed_dim,
            num_heads=8,
            ffn_dim=embed_dim * 4,
            num_layers=8,
            dropout=0.1,
            depthwise_conv_kernel_size=31
        )
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks * 2)

    def forward(self, x_stft_mag, x_audio):
        B, C, F, T = x_stft_mag.shape
        x_stft_mag = x_stft_mag.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x = self.input_proj_stft(x_stft_mag)
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        x, _ = self.model(x, lengths)
        x = self.output_proj(x)
        current_T = x.shape[1]
        x = x.view(B, current_T, self.out_masks * 2, F).permute(0, 2, 3, 1)
        return x

class MultiResolutionComplexSTFTLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super(MultiResolutionComplexSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        for i, win_len in enumerate(win_lengths):
            self.register_buffer(f'window_{i}', torch.hann_window(win_len), persistent=False)

    def forward(self, y_pred, y_true):
        complex_loss_total = 0.0

        if y_pred.dim() == 3:
            y_pred = y_pred.reshape(y_pred.size(0), -1)
            y_true = y_true.reshape(y_true.size(0), -1)

        for i, (n_fft, hop_length, win_length) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f'window_{i}')

            stft_pred = torch.stft(y_pred, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            stft_true = torch.stft(y_true, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

            real_loss = F.mse_loss(stft_pred.real, stft_true.real)
            imag_loss = F.mse_loss(stft_pred.imag, stft_true.imag)
            
            complex_loss_total += (real_loss + imag_loss)

        return complex_loss_total

def loss_fn(pred_output,
            mixture_spec,
            target_vocal_audio,
            target_instr_audio,
            target_vocal_spec,
            target_instr_spec,
            stft_params_for_istft,
            multi_res_complex_loss_calculator):
    device = pred_output.device

    B, _, F_dim, T = pred_output.shape
    pred_masks = pred_output.view(B, 2, 4, F_dim, T)
    pred_masks_real = pred_masks[:, 0]
    pred_masks_imag = pred_masks[:, 1]
    vL_cmask = pred_masks_real[:, 0] + 1j * pred_masks_imag[:, 0]
    vR_cmask = pred_masks_real[:, 1] + 1j * pred_masks_imag[:, 1]
    iL_cmask = pred_masks_real[:, 2] + 1j * pred_masks_imag[:, 2]
    iR_cmask = pred_masks_real[:, 3] + 1j * pred_masks_imag[:, 3]
    vL_cmask, vR_cmask = vL_cmask.unsqueeze(1), vR_cmask.unsqueeze(1)
    iL_cmask, iR_cmask = iL_cmask.unsqueeze(1), iR_cmask.unsqueeze(1)
    v_spec_pred = torch.cat([vL_cmask * mixture_spec[:, 0:1],
                             vR_cmask * mixture_spec[:, 1:2]], dim=1)
    i_spec_pred = torch.cat([iL_cmask * mixture_spec[:, 0:1],
                             iR_cmask * mixture_spec[:, 1:2]], dim=1)

    spec_vocal_loss = F.l1_loss(v_spec_pred.real, target_vocal_spec.real) + \
                      F.l1_loss(v_spec_pred.imag, target_vocal_spec.imag)
    spec_instr_loss = F.l1_loss(i_spec_pred.real, target_instr_spec.real) + \
                      F.l1_loss(i_spec_pred.imag, target_instr_spec.imag)
    spectrogram_loss = spec_vocal_loss + spec_instr_loss

    n_fft = stft_params_for_istft['n_fft']
    hop_length = stft_params_for_istft['hop_length']
    window = stft_params_for_istft['window'].to(device)
    recon_len = target_vocal_audio.shape[-1]

    B, C, freq, T_spec = v_spec_pred.shape
    v_spec_pred_reshaped = v_spec_pred.reshape(B * C, freq, T_spec)
    i_spec_pred_reshaped = i_spec_pred.reshape(B * C, freq, T_spec)

    pred_vocal_audio = torch.istft(
        v_spec_pred_reshaped, n_fft=n_fft, hop_length=hop_length,
        window=window, center=True, length=recon_len
    ).reshape(B, C, -1)
    pred_instr_audio = torch.istft(
        i_spec_pred_reshaped, n_fft=n_fft, hop_length=hop_length,
        window=window, center=True, length=recon_len
    ).reshape(B, C, -1)

    vocal_loss = multi_res_complex_loss_calculator(pred_vocal_audio, target_vocal_audio)
    instr_loss = multi_res_complex_loss_calculator(pred_instr_audio, target_instr_audio)
    audio_loss = vocal_loss + instr_loss

    total_loss = 0.5 * spectrogram_loss + 0.5 * audio_loss

    return total_loss

class Dataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_length=88200, segment=True):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment = segment

        self.n_fft = 4096
        self.hop_length = 1024
        self.window = torch.hann_window(self.n_fft)

        self.vocal_paths = []
        self.other_paths = []

        track_dirs = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]

        print("Scanning track folders...")
        for td in tqdm(track_dirs, desc="Scanning tracks"):
            vocal_path = self._find_audio_file(td, 'vocals')
            other_path = self._find_audio_file(td, 'other')

            if vocal_path:
                self.vocal_paths.append(vocal_path)
            if other_path:
                self.other_paths.append(other_path)

        if not self.vocal_paths or not self.other_paths:
            raise ValueError("Dataset must contain both vocal and 'other' stems.")

        self.size = 50000

    def _find_audio_file(self, directory, base_name):
        for ext in ['.wav', '.flac']:
            path = os.path.join(directory, base_name + ext)
            if os.path.exists(path):
                return path
        return None

    def _preprocess_audio(self, audio, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        return audio

    def _load_audio(self, filepath):
        audio, sr = torchaudio.load(filepath)
        return self._preprocess_audio(audio, sr)

    def _load_vocal(self, path):
        return self._load_audio(path)

    def _load_instrumental(self, path):
        return self._load_audio(path)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        vocal_path = random.choice(self.vocal_paths)
        other_path = random.choice(self.other_paths)

        vocal_audio = self._load_vocal(vocal_path)
        instr_audio = self._load_instrumental(other_path)

        min_length = min(vocal_audio.shape[1], instr_audio.shape[1])
        if min_length < self.segment_length:
            raise ValueError(f"Encountered an audio file shorter than the segment length. Min length: {min_length}")

        if self.segment:
            start = random.randint(0, min_length - self.segment_length)
            end = start + self.segment_length
            vocal_seg = vocal_audio[:, start:end]
            instr_seg = instr_audio[:, start:end]
        else:
            vocal_seg = vocal_audio
            instr_seg = instr_audio

        mixture_seg = vocal_seg + instr_seg

        mixture_spec = torch.stft(mixture_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=self.window, return_complex=True, center=True)
        
        target_vocal_spec = torch.stft(vocal_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=self.window, return_complex=True, center=True)
        
        target_instr_spec = torch.stft(instr_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=self.window, return_complex=True, center=True)

        return mixture_spec, vocal_seg, instr_seg, mixture_seg, target_vocal_spec, target_instr_spec

def train(model, dataloader, optimizer, loss_fn, device, checkpoint_steps, args, checkpoint_path=None, window=None, reset_optimizer=False):
    model.to(device)
    step = 0
    avg_loss = 0.0
    checkpoint_files = []

    stft_params_for_istft = {
        'n_fft': 4096,
        'hop_length': 1024,
        'window': window.to(device)
    }

    multi_res_complex_loss_calculator = MultiResolutionComplexSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192]
    ).to(device)

    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        step = checkpoint_data['step']
        avg_loss = checkpoint_data['avg_loss']

        if reset_optimizer:
            print(f"Resuming training from step {step}. MODEL LOADED, OPTIMIZER RESET.")
        else:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            print(f"Resuming training from step {step} with average loss {avg_loss:.4f}. MODEL AND OPTIMIZER LOADED.")

    progress_bar = tqdm(initial=step, total=None)
    model.train()
    while True:
        for batch in dataloader:
            mixture_spec, vocal_audio, instr_audio, mixture_audio, target_vocal_spec, target_instr_spec = batch
            
            mixture_mag = torch.abs(mixture_spec).to(device)
            mixture_spec = mixture_spec.to(device)
            vocal_audio = vocal_audio.to(device)
            instr_audio = instr_audio.to(device)
            mixture_audio = mixture_audio.to(device)
            target_vocal_spec = target_vocal_spec.to(device)
            target_instr_spec = target_instr_spec.to(device)

            optimizer.zero_grad()
            pred_masks = model(mixture_mag, mixture_audio)

            loss = loss_fn(pred_masks,
                           mixture_spec,
                           vocal_audio,
                           instr_audio,
                           target_vocal_spec,
                           target_instr_spec,
                           stft_params_for_istft,
                           multi_res_complex_loss_calculator)
            
            if torch.isnan(loss).any():
                print("NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            step += 1
            progress_bar.update(1)
            current_lr = optimizer.param_groups[0]['lr']
            desc = f"Step {step} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f}"
            progress_bar.set_description(desc)

            if step % checkpoint_steps == 0:
                checkpoint_filename = f"checkpoint_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                }, checkpoint_filename)
                checkpoint_files.append(checkpoint_filename)
                if len(checkpoint_files) > 3:
                    oldest_checkpoint = checkpoint_files.pop(0)
                    if os.path.exists(oldest_checkpoint):
                        os.remove(oldest_checkpoint)

def inference(model, checkpoint_path, input_path, output_instrumental_path, output_vocal_path,
              chunk_size=352800, overlap=88200, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval().to(device)

    input_audio, sr = torchaudio.load(input_path)
    if sr != 44100:
        raise ValueError(f"Input audio must be 44100Hz, but got {sr}Hz.")
    if input_audio.shape[0] == 1:
        input_audio = input_audio.repeat(2, 1)
    elif input_audio.shape[0] != 2:
        raise ValueError("Input audio must be mono or stereo.")
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

    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
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
    torchaudio.save(output_vocal_path, vocals.cpu(), sr)
    torchaudio.save(output_instrumental_path, instrumentals.cpu(), sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for instrumental separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='train', help='Path to training dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint_steps', type=int, default=2000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_file', type=str, default=None, help='Path to input audio file for inference (wav or flac)')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--segment_length', type=int, default=352800, help='Segment length for training')
    parser.add_argument('--reset_optimizer', action='store_true', help='Reset optimizer state when resuming from a checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel()
    optimizer = Prodigy(model.parameters(), lr=1.0)

    if args.train:
        train_dataset = Dataset(root_dir=args.data_dir,
                                      segment_length=args.segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=16, pin_memory=True, persistent_workers=True)
        train(model, train_dataloader, optimizer, loss_fn, device, args.checkpoint_steps, args, checkpoint_path=args.checkpoint_path, window=window, reset_optimizer=args.reset_optimizer)
    elif args.infer:
        if args.input_file is None:
            print("Please specify an input audio file for inference using --input_file")
            return
        inference(model, args.checkpoint_path, args.input_file, args.output_instrumental, args.output_vocal, device=device)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
