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
import re
import warnings

warnings.filterwarnings("ignore")

def rotary_emb(dim, seq_len, base=10000.0, device=None):
    if dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even.")
    device = device or torch.device("cpu")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("l,d->ld", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    cos = cos[:T, :D].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, :D].unsqueeze(0).unsqueeze(0)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos[..., ::2] - x_odd * sin[..., ::2]
    out_odd = x_even * sin[..., ::2] + x_odd * cos[..., ::2]
    out = torch.empty_like(x)
    out[..., ::2] = out_even
    out[..., 1::2] = out_odd
    return out

class MaskedConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        mask = torch.zeros_like(self.weight.data)
        cx = cy = cz = 1
        mask[:, :, cx, cy, cz] = 1.0
        mask[:, :, cx - 1, cy, cz] = 1.0
        mask[:, :, cx + 1, cy, cz] = 1.0
        mask[:, :, cx, cy - 1, cz] = 1.0
        mask[:, :, cx, cy + 1, cz] = 1.0
        mask[:, :, cx, cy, cz - 1] = 1.0
        mask[:, :, cx, cy, cz + 1] = 1.0
        self.register_buffer("mask", mask, persistent=False)
    def forward(self, x):
        w = self.weight * self.mask
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class RandomGraphMixer3D(nn.Module):
    def __init__(self, c_in, c_out, grid, num_rand, seed=0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.grid = grid
        self.num_rand = num_rand
        X, Y, Z = grid
        N = X * Y * Z
        g = torch.Generator()
        g.manual_seed(seed)
        rand_indices = torch.randint(0, N, (N, num_rand), generator=g)
        self.register_buffer("rand_indices", rand_indices, persistent=False)
        self.weight = nn.Parameter(torch.zeros(c_out, c_in, num_rand))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(c_out))
    def forward(self, x):
        BT, C, X, Y, Z = x.shape
        N = X * Y * Z
        x_nodes = x.view(BT, C, N)
        idx = self.rand_indices
        gathered = x_nodes[:, :, idx]
        gathered = gathered.permute(0, 1, 3, 2).contiguous()
        out = torch.einsum('ocr, bcrn -> bon', self.weight, gathered) + self.bias.view(1, -1, 1)
        out = out.view(BT, self.c_out, X, Y, Z)
        return out

class HybridProjection3D(nn.Module):
    def __init__(self, in_dim, out_dim, grid, num_rand=8):
        super().__init__()
        self.grid = grid
        X, Y, Z = grid
        if in_dim % (X * Y * Z) != 0 or out_dim % (X * Y * Z) != 0:
            raise ValueError("in_dim and out_dim must be divisible by X*Y*Z")
        self.c_in = in_dim // (X * Y * Z)
        self.c_out = out_dim // (X * Y * Z)
        self.local = MaskedConv3d(self.c_in, self.c_out, bias=True)
        self.rand = RandomGraphMixer3D(self.c_in, self.c_out, grid, num_rand=num_rand, seed=42)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        B, T, D = x.shape
        X, Y, Z = self.grid
        xt = x.view(B * T, self.c_in, X, Y, Z)
        y_local = self.local(xt)
        y_rand = self.rand(xt)
        a = torch.sigmoid(self.alpha)
        yt = a * y_local + (1 - a) * y_rand
        y = yt.view(B, T, self.c_out * X * Y * Z)
        return y

class CogAttention(nn.Module):
    def forward(self, logits):
        return torch.tanh(logits)

class CogMultiheadAttention3D(nn.Module):
    def __init__(self, d_model, n_heads, grid, rope_base=10000.0, dropout=0.0, num_rand=8):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_proj = HybridProjection3D(d_model, d_model, grid, num_rand=num_rand)
        self.k_proj = HybridProjection3D(d_model, d_model, grid, num_rand=num_rand)
        self.v_proj = HybridProjection3D(d_model, d_model, grid, num_rand=num_rand)
        self.o_proj = HybridProjection3D(d_model, d_model, grid, num_rand=num_rand)
        self.dropout = nn.Dropout(dropout)
        self.rope_base = rope_base
        self.cog = CogAttention()
    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        cos, sin = rotary_emb(self.head_dim, T, base=self.rope_base, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask
        attn_weights = self.cog(attn_logits)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        norm_factor = attn_weights.abs().sum(dim=-1, keepdim=True) + 1e-6
        context = context / norm_factor
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(context)
        return out

class MLP3D(nn.Module):
    def __init__(self, d_model, expansion, grid, dropout=0.0, num_rand=8):
        super().__init__()
        d_hidden = d_model * expansion
        self.fc1 = HybridProjection3D(d_model, d_hidden, grid, num_rand=num_rand)
        self.act = nn.GELU()
        self.fc2 = HybridProjection3D(d_hidden, d_model, grid, num_rand=num_rand)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock3D(nn.Module):
    def __init__(self, d_model, n_heads, grid, mlp_expansion=4, dropout=0.0, num_rand=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CogMultiheadAttention3D(d_model, n_heads, grid, dropout=dropout, num_rand=num_rand)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP3D(d_model, mlp_expansion, grid, dropout=dropout, num_rand=num_rand)
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer3D(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, grid, mlp_expansion=4, dropout=0.0, num_rand=8):
        super().__init__()
        X, Y, Z = grid
        if d_model % (X * Y * Z) != 0:
            raise ValueError("d_model must be divisible by X*Y*Z for 3D lattice factorization")
        self.layers = nn.ModuleList([
            TransformerBlock3D(d_model, n_heads, grid, mlp_expansion, dropout, num_rand=num_rand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)
        return x

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049, embed_dim=512, depth=6, heads=8):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim
        self.input_proj_stft = nn.Linear(freq_bins * in_channels * 2, embed_dim)
        grid = (8, 8, 8)
        self.model = Transformer3D(
            d_model=embed_dim,
            n_heads=heads,
            n_layers=depth,
            grid=grid,
            mlp_expansion=4,
            dropout=0.0,
            num_rand=8
        )
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks * 2)
    def forward(self, x_stft, x_audio):
        x_stft = torch.cat([x_stft.real, x_stft.imag], dim=1)
        B, C, F, T = x_stft.shape
        x_stft = x_stft.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x = self.input_proj_stft(x_stft)
        x = self.model(x)
        x = torch.tanh(x)
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

        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)

        B, C, L = y_pred.shape
        y_pred_flat = y_pred.reshape(B * C, L)
        y_true_flat = y_true.reshape(B * C, L)

        for i, (n_fft, hop_length, win_length) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f'window_{i}')
            window = window.to(y_pred_flat.device)

            stft_pred = torch.stft(y_pred_flat, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=window, return_complex=True, center=True)
            stft_true = torch.stft(y_true_flat, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=window, return_complex=True, center=True)

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
    pred_output_reshaped = pred_output.view(B, 2, 4, F_dim, T)
    pred_real = pred_output_reshaped[:, 0]
    pred_imag = pred_output_reshaped[:, 1]

    pred_masks_real = pred_real[:, :4]
    pred_masks_imag = pred_imag[:, :4]

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
        while True:
            try:
                vocal_path = random.choice(self.vocal_paths)
                other_path = random.choice(self.other_paths)

                vocal_audio = self._load_vocal(vocal_path)
                instr_audio = self._load_instrumental(other_path)

                min_length = min(vocal_audio.shape[1], instr_audio.shape[1])
                if self.segment and min_length < self.segment_length:
                    continue

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
            except Exception as e:
                print(f"Error loading item, trying next. Error: {e}")
                continue

def calculate_sdr(pred, target, epsilon=1e-8):
    noise = pred - target
    s_power = torch.sum(target**2, dim=-1, keepdim=True)
    n_power = torch.sum(noise**2, dim=-1, keepdim=True)
    sdr = 10 * torch.log10(s_power / (n_power + epsilon) + epsilon)
    return sdr.mean().item()

def validate(model, test_dir, device, chunk_size, overlap):
    model.eval()
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        print(f"Test directory not found or is empty: {test_dir}")
        return 0.0, 0.0, 0.0
        
    track_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    if not track_dirs:
        print(f"No subdirectories found in test directory: {test_dir}")
        return 0.0, 0.0, 0.0

    total_vocal_sdr, total_instr_sdr, count = 0.0, 0.0, 0

    with tqdm(track_dirs, desc="Validating", leave=False) as pbar:
        for track_dir in pbar:
            try:
                vocal_path = next(os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.startswith('vocals') and f.endswith(('.wav', '.flac')))
                other_path = next(os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.startswith('other') and f.endswith(('.wav', '.flac')))
                
                gt_vocals, sr_v = torchaudio.load(vocal_path)
                gt_instr, sr_i = torchaudio.load(other_path)

                if sr_v != 44100 or sr_i != 44100: continue
                
                min_len = min(gt_vocals.shape[1], gt_instr.shape[1])
                gt_vocals, gt_instr = gt_vocals[:, :min_len], gt_instr[:, :min_len]

                if gt_vocals.shape[0] == 1: gt_vocals = gt_vocals.repeat(2, 1)
                if gt_instr.shape[0] == 1: gt_instr = gt_instr.repeat(2, 1)

                mixture = (gt_vocals + gt_instr).to(device)
                gt_vocals, gt_instr = gt_vocals.to(device), gt_instr.to(device)

                with torch.no_grad():
                    pred_vocals, pred_instr = inference(model, None, mixture, None, None, chunk_size, overlap, device, return_tensors=True)

                vocal_sdr = calculate_sdr(pred_vocals, gt_vocals)
                instr_sdr = calculate_sdr(pred_instr, gt_instr)
                
                total_vocal_sdr += vocal_sdr
                total_instr_sdr += instr_sdr
                count += 1
                
                pbar.set_postfix({'vocal_sdr': f"{vocal_sdr:.2f}", 'instr_sdr': f"{instr_sdr:.2f}"})
            except (StopIteration, Exception) as e:
                continue

    if count == 0:
        print("No valid test files processed.")
        return 0.0, 0.0, 0.0

    avg_vocal_sdr = total_vocal_sdr / count
    avg_instr_sdr = total_instr_sdr / count
    avg_combined_sdr = (avg_vocal_sdr + avg_instr_sdr) / 2
    return avg_vocal_sdr, avg_instr_sdr, avg_combined_sdr

def train(model, dataloader, optimizer, loss_fn, device, checkpoint_steps, args, checkpoint_path=None, window=None, reset_optimizer=False):
    model.to(device)
    step = 0
    avg_loss = 0.0
    
    best_sdr = -float('inf')
    if os.path.exists('best_ckpts') and os.listdir('best_ckpts'):
        sdr_values = [float(re.search(r"sdr_([\d\.]+)\.pt", f).group(1)) for f in os.listdir('best_ckpts') if re.search(r"sdr_([\d\.]+)\.pt", f)]
        if sdr_values:
            best_sdr = max(sdr_values)

    stft_params_for_istft = {
        'n_fft': 4096, 'hop_length': 1024, 'window': window.to(device)
    }
    multi_res_complex_loss_calculator = MultiResolutionComplexSTFTLoss(
        fft_sizes=[1024, 2048, 8192], hop_sizes=[256, 512, 2048], win_lengths=[1024, 2048, 8192]
    ).to(device)

    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        step = checkpoint_data.get('step', 0)
        avg_loss = checkpoint_data.get('avg_loss', 0.0)
        
        if not reset_optimizer and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            print(f"Resuming training from step {step}. MODEL AND OPTIMIZER LOADED.")
        else:
            print(f"Resuming training from step {step}. MODEL LOADED, OPTIMIZER RESET.")

    progress_bar = tqdm(initial=step, total=None, dynamic_ncols=True)
    
    while True:
        model.train()
        for batch in dataloader:
            mixture_spec, vocal_audio, instr_audio, mixture_audio, target_vocal_spec, target_instr_spec = batch
            
            mixture_spec, vocal_audio, instr_audio, mixture_audio = mixture_spec.to(device, non_blocking=True), vocal_audio.to(device, non_blocking=True), instr_audio.to(device, non_blocking=True), mixture_audio.to(device, non_blocking=True)
            target_vocal_spec, target_instr_spec = target_vocal_spec.to(device, non_blocking=True), target_instr_spec.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_masks = model(mixture_spec, mixture_audio)
            loss = loss_fn(pred_masks, mixture_spec, vocal_audio, instr_audio, target_vocal_spec, target_instr_spec, stft_params_for_istft, multi_res_complex_loss_calculator)
            
            if torch.isnan(loss).any(): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_loss = 0.999 * avg_loss + 0.001 * loss.item() if step > 0 else loss.item()
            step += 1
            progress_bar.update(1)
            progress_bar.set_description(f"Step {step} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f} - Best SDR: {best_sdr:.4f}")

            if step > 0 and step % checkpoint_steps == 0:
                checkpoint_payload = {
                    'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss
                }

                reg_checkpoint_path = f"ckpts/checkpoint_step_{step}.pt"
                torch.save(checkpoint_payload, reg_checkpoint_path)

                regular_checkpoints = sorted(
                    [os.path.join('ckpts', f) for f in os.listdir('ckpts') if f.endswith('.pt')],
                    key=os.path.getmtime
                )
                if len(regular_checkpoints) > 3:
                    os.remove(regular_checkpoints[0])

                avg_vocal_sdr, avg_instr_sdr, avg_combined_sdr = validate(model, args.test_dir, device, args.segment_length, overlap=88200)
                
                print(f"\nValidation Step {step}: Vocal SDR: {avg_vocal_sdr:.4f}, Instr SDR: {avg_instr_sdr:.4f}, Combined SDR: {avg_combined_sdr:.4f}")

                best_sdr_checkpoints = []
                if os.path.exists('best_ckpts'):
                    for f in os.listdir('best_ckpts'):
                        match = re.search(r"sdr_([\d\.]+)\.pt", f)
                        if match:
                            sdr = float(match.group(1))
                            best_sdr_checkpoints.append((sdr, os.path.join('best_ckpts', f)))
                
                best_sdr_checkpoints.sort(key=lambda x: x[0])

                current_best_sdr = best_sdr_checkpoints[-1][0] if best_sdr_checkpoints else -float('inf')
                if avg_combined_sdr > current_best_sdr:
                    best_sdr = avg_combined_sdr
                    
                    for _, path in best_sdr_checkpoints:
                        try:
                            os.remove(path)
                        except Exception:
                            pass

                    best_ckpt_filename = f"best_ckpts/checkpoint_step_{step}_sdr_{avg_combined_sdr:.4f}.pt"
                    torch.save(checkpoint_payload, best_ckpt_filename)
                    print(f"New best SDR! Saved checkpoint: {best_ckpt_filename}\n")
                else:
                    print(f"SDR did not improve. Best SDR remains {best_sdr:.4f}\n")
                
                model.train()

def find_latest_checkpoint(folder='ckpts'):
    if not os.path.exists(folder) or not os.listdir(folder):
        return None

    checkpoints = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pt')]
    if not checkpoints:
        return None

    def get_step_from_path(path):
        filename = os.path.basename(path)
        match = re.search(r'step_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0

    latest_checkpoint = max(checkpoints, key=get_step_from_path)
    return latest_checkpoint

def find_best_sdr_checkpoint(folder='best_ckpts'):
    if not os.path.exists(folder) or not os.listdir(folder):
        return None

    best_sdr = -float('inf')
    best_ckpt = None
    for f in os.listdir(folder):
        if f.endswith('.pt'):
            match = re.search(r"sdr_([\d\.]+)\.pt", f)
            if match:
                sdr = float(match.group(1))
                if sdr > best_sdr:
                    best_sdr = sdr
                    best_ckpt = os.path.join(folder, f)
    return best_ckpt

def inference(model, checkpoint_path, input_data, output_instrumental_path, output_vocal_path,
              chunk_size=485100, overlap=88200, device='cpu', return_tensors=False):
    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    
    model.eval().to(device)

    if isinstance(input_data, str):
        input_audio, sr = torchaudio.load(input_data)
        if sr != 44100: raise ValueError(f"Input audio must be 44100Hz, but got {sr}Hz.")
    else:
        input_audio = input_data

    if input_audio.shape[0] == 1: input_audio = input_audio.repeat(2, 1)
    elif input_audio.shape[0] != 2: raise ValueError("Input audio must be mono or stereo.")
    input_audio = input_audio.to(device)

    total_length = input_audio.shape[1]
    vocals = torch.zeros_like(input_audio)
    instrumentals = torch.zeros_like(input_audio)
    sum_fade_windows = torch.zeros(total_length, device=device)

    n_fft, hop_length = 4096, 1024
    window = torch.hann_window(n_fft).to(device)
    step_size = chunk_size - overlap

    with tqdm(total=math.ceil(total_length / step_size), desc="Processing audio", leave=False) as pbar:
        for i in range(0, total_length, step_size):
            start, end = i, min(i + chunk_size, total_length)
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

            if i > 0:
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

    if return_tensors:
        return vocals.clamp(-1.0, 1.0), instrumentals.clamp(-1.0, 1.0)
    else:
        torchaudio.save(output_vocal_path, vocals.cpu().clamp(-1.0, 1.0), 44100)
        torchaudio.save(output_instrumental_path, instrumentals.cpu().clamp(-1.0, 1.0), 44100)

def main():
    parser = argparse.ArgumentParser(description='Train or run inference on a source separation model.')
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--infer', action='store_true', help='Run inference.')
    parser.add_argument('--data_dir', type=str, default='train', help='Path to the training dataset.')
    parser.add_argument('--test_dir', type=str, default='test', help='Path to the test dataset for validation.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--checkpoint_steps', type=int, default=4000, help='Save a checkpoint every X steps.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Specific checkpoint path to resume training or for inference. Overrides automatic selection.')
    parser.add_argument('--input_file', type=str, default=None, help='Path to the input audio file for inference.')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path for the output instrumental file.')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path for the output vocal file.')
    parser.add_argument('--segment_length', type=int, default=485100, help='Audio segment length for training and inference chunk size.')
    parser.add_argument('--reset_optimizer', action='store_true', help='Reset optimizer state when resuming from a checkpoint.')
    args = parser.parse_args()

    os.makedirs('ckpts', exist_ok=True)
    os.makedirs('best_ckpts', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel() 
    optimizer = Prodigy(model.parameters(), lr=1.0)

    if args.train:
        checkpoint_to_load = args.checkpoint_path
        if not checkpoint_to_load:
            checkpoint_to_load = find_latest_checkpoint('ckpts')
            if checkpoint_to_load:
                print(f"Automatically resuming from latest checkpoint: {checkpoint_to_load}")

        train_dataset = Dataset(root_dir=args.data_dir, segment_length=args.segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
        train(model, train_dataloader, optimizer, loss_fn, device, args.checkpoint_steps, args, checkpoint_path=checkpoint_to_load, window=window, reset_optimizer=args.reset_optimizer)
    
    elif args.infer:
        if not args.input_file:
            print("Error: --input_file is required for inference.")
            return

        checkpoint_to_load = args.checkpoint_path
        if not checkpoint_to_load:
            checkpoint_to_load = find_best_sdr_checkpoint('best_ckpts')
            if checkpoint_to_load:
                print(f"No checkpoint specified. Automatically using best SDR checkpoint: {checkpoint_to_load}")
            else:
                print("Error: No checkpoint specified with --checkpoint_path and no best SDR checkpoint was found in 'best_ckpts/'.")
                return
        else:
            print(f"Using specified checkpoint: {checkpoint_to_load}")

        inference(model, checkpoint_to_load, args.input_file, args.output_instrumental, args.output_vocal,
                  chunk_size=args.segment_length, device=device)
    
    else:
        print("Please specify either --train or --infer.")

if __name__ == '__main__':
    main()