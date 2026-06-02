import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from adam_atan2_pytorch import AdamAtan2
from torch.cuda.amp import autocast, GradScaler
import random
import math
import re
import warnings

warnings.filterwarnings("ignore")

STEMS = ['vocals', 'other']

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class PoPE(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim = dim

        inv_freqs = 10000 ** -(torch.arange(0, dim, 1).float() / dim)
        self.register_buffer('inv_freqs', inv_freqs)

        self.bias = nn.Parameter(torch.zeros(heads, dim))

    def forward(self, seq_len, device):
        pos = torch.arange(seq_len, device=device, dtype=self.inv_freqs.dtype)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freqs)

        bias = self.bias.clamp(-2 * math.pi, 0)

        freqs_with_bias = freqs.unsqueeze(0) + bias.unsqueeze(1)

        return freqs, freqs_with_bias

def apply_pope(q, k, freqs, freqs_with_bias):
    q_mag = F.softplus(q.float())
    k_mag = F.softplus(k.float())

    q_phase = freqs.unsqueeze(0).unsqueeze(0).float()
    k_phase = freqs_with_bias.unsqueeze(0).float()

    q_complex = torch.polar(q_mag, q_phase)
    k_complex = torch.polar(k_mag, k_phase)

    q_embed = torch.view_as_real(q_complex).flatten(-2)
    k_embed = torch.view_as_real(k_complex).flatten(-2)

    return q_embed.type_as(q), k_embed.type_as(k)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.pope = PoPE(self.head_dim, heads)

    def forward(self, x):
        B, S, C = x.shape
        H, D = self.heads, self.head_dim

        q = self.wq(x).view(B, S, H, D).transpose(1, 2)
        k = self.wk(x).view(B, S, H, D).transpose(1, 2)
        v = self.wv(x).view(B, S, H, D).transpose(1, 2)

        freqs, bias = self.pope(S, x.device)

        q, k = apply_pope(q, k, freqs, bias)

        out = F.scaled_dot_product_attention(q, k, v, scale=(2*D)**-0.5)

        out = out.transpose(1, 2).reshape(B, S, C)

        return self.out_proj(out)

class GatedFF(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)

        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.ff = GatedFF(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class DualPathEncoder(nn.Module):
    def __init__(self, dim, depth, heads, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.time_layers = nn.ModuleList([EncoderLayer(dim, heads) for i in range(depth)])
        self.freq_layers = nn.ModuleList([EncoderLayer(dim, heads) for i in range(depth)])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        B, F_b, T, E = x.shape
        for t_layer, f_layer in zip(self.time_layers, self.freq_layers):
            x = x.contiguous().view(B * F_b, T, E)

            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(t_layer, x, use_reentrant=True)
            else:
                x = t_layer(x)
            x = x.view(B, F_b, T, E)

            x = x.transpose(1, 2).contiguous().view(B * T, F_b, E)
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(f_layer, x, use_reentrant=True)
            else:
                x = f_layer(x)
            x = x.view(B, T, F_b, E).transpose(1, 2)
        return self.norm(x)

class NeuralModel(nn.Module):
    def __init__(
        self,
        in_channels=2,
        sources=len(STEMS),
        freq_bins=2049,
        embed_dim=384,
        depth=8,
        heads=8,
        hop_length=1024,
        window_size=4096,
        use_checkpoint=False,
        downsample=12
    ):
        super().__init__()

        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.freq_bins = freq_bins
        self.downsample = downsample
        
        unshuffled_channels = in_channels * 2 * downsample
        self.input_proj = nn.Conv2d(
            in_channels=unshuffled_channels,
            out_channels=embed_dim,
            kernel_size=1
        )
        
        self.norm = RMSNorm(embed_dim)

        self.model = DualPathEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            use_checkpoint=use_checkpoint
        )

        upsample_dim = self.out_masks * 2 * downsample
        self.proj_to_pixel_shuffle = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=upsample_dim,
            kernel_size=1
        )

    def forward(self, x_stft):
        x = torch.cat([x_stft.real, x_stft.imag], dim=1)

        rem = x.shape[2] % self.downsample
        if rem != 0:
            pad_amount = self.downsample - rem
            x = F.pad(x, (0, 0, 0, pad_amount))

        B, C, F_len, T = x.shape
        x = x.view(B, C, F_len // self.downsample, self.downsample, T)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, C * self.downsample, F_len // self.downsample, T)

        x = self.input_proj(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.model(x)

        x = x.permute(0, 3, 1, 2)
        x = self.proj_to_pixel_shuffle(x)
        
        B, C, F_down, T = x.shape
        x = x.view(B, self.out_masks * 2, self.downsample, F_down, T)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, self.out_masks * 2, F_down * self.downsample, T)

        if x.shape[2] < self.freq_bins:
            x = F.pad(x, (0, 0, 0, self.freq_bins - x.shape[2]))
        elif x.shape[2] > self.freq_bins:
            x = x[:, :, :self.freq_bins, :]

        return torch.tanh(x)

class MultiResolutionComplexSTFTLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super(MultiResolutionComplexSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        inv = torch.tensor([1.0 / n for n in fft_sizes])
        self.register_buffer('weights', inv / inv.sum(), persistent=False)
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

            stft_pred = torch.stft(y_pred_flat, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=window, return_complex=True, center=True)
            stft_true = torch.stft(y_true_flat, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=window, return_complex=True, center=True)

            real_loss = F.l1_loss(stft_pred.real, stft_true.real)
            imag_loss = F.l1_loss(stft_pred.imag, stft_true.imag)

            complex_loss_total += (real_loss + imag_loss) * self.weights[i]

        return complex_loss_total

def loss_fn(pred_output,
            mixture_spec,
            target_audios,
            stft_params_for_istft,
            multi_res_complex_loss_calculator):
    device = pred_output.device

    B, _, F_dim, T = pred_output.shape
    num_stems = target_audios.shape[1]
    pred_output_reshaped = pred_output.contiguous().view(B, 2, num_stems * 2, F_dim, T)
    pred_real = pred_output_reshaped[:, 0]
    pred_imag = pred_output_reshaped[:, 1]

    n_fft = stft_params_for_istft['n_fft']
    hop_length = stft_params_for_istft['hop_length']
    window = stft_params_for_istft['window'].to(device)
    recon_len = target_audios.shape[-1]

    total_loss = 0.0
    for i in range(num_stems):
        cmask = pred_real[:, 2*i:2*i+2] + 1j * pred_imag[:, 2*i:2*i+2]
        stem_spec_pred = cmask * mixture_spec

        B_s, C_s, freq, T_spec = stem_spec_pred.shape
        stem_spec_pred_reshaped = stem_spec_pred.reshape(B_s * C_s, freq, T_spec)

        pred_audio = torch.istft(
            stem_spec_pred_reshaped, n_fft=n_fft, hop_length=hop_length,
            window=window, center=True, length=recon_len
        ).reshape(B_s, C_s, -1)

        total_loss += multi_res_complex_loss_calculator(pred_audio, target_audios[:, i])
        total_loss += F.l1_loss(pred_audio, target_audios[:, i])

    return total_loss

class Dataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_length=264600, segment=True):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment = segment

        self.n_fft = 4096
        self.hop_length = 1024
        self.window = torch.hann_window(self.n_fft)

        self.stems = STEMS
        self.tracks = {stem: [] for stem in self.stems}

        track_dirs = [os.path.join(root_dir, track) for track in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, track))]

        print("Scanning and caching track metadata...")
        for td in tqdm(track_dirs, desc="Caching tracks"):
            for stem in self.stems:
                path = self._find_audio_file(td, stem)
                if path:
                    info = sf.info(path)
                    self.tracks[stem].append({'path': path, 'frames': info.frames, 'sr': info.samplerate})

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

    def _load_chunk(self, track_info):
        path = track_info['path']
        total_frames = track_info['frames']
        sr = track_info['sr']

        target_length = self.segment_length if self.segment else total_frames

        if total_frames <= target_length:
            audio_np, sr = sf.read(path, dtype='float32')
            audio = torch.from_numpy(audio_np)
            audio = audio.unsqueeze(0) if audio.dim() == 1 else audio.t()

            audio = self._preprocess_audio(audio, sr)
            pad_size = target_length - audio.shape[1]
            if pad_size > 0:
                audio = F.pad(audio, (0, pad_size))
        else:
            start_frame = random.randint(0, total_frames - target_length)
            audio_np, sr = sf.read(path, start=start_frame, frames=target_length, dtype='float32')
            audio = torch.from_numpy(audio_np)
            audio = audio.unsqueeze(0) if audio.dim() == 1 else audio.t()

            audio = self._preprocess_audio(audio, sr)

        return audio

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        while True:
            try:
                target_audios = []
                for stem in self.stems:
                    if self.tracks[stem]:
                        track_info = random.choice(self.tracks[stem])
                        target_audios.append(self._load_chunk(track_info))
                    else:
                        target_audios.append(torch.zeros(2, self.segment_length))

                mixture_seg = torch.stack(target_audios).sum(dim=0)
                mixture_spec = torch.stft(mixture_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                          window=self.window, return_complex=True, center=True)

                return mixture_spec, torch.stack(target_audios)
            except Exception:
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
        return 0.0

    track_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not track_dirs:
        print(f"No valid track directories found in: {test_dir}")
        return 0.0

    total_sdr, count = 0.0, 0
    num_stems = len(STEMS)
    total_stems_sdr = [0.0] * num_stems

    with tqdm(track_dirs, desc="Validating", leave=False) as pbar:
        for track_dir in pbar:
            try:
                gt_audios = []
                for stem in STEMS:
                    path = next(os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.startswith(stem) and f.endswith(('.wav', '.flac')))
                    audio_np, sr = sf.read(path, dtype='float32')
                    audio = torch.from_numpy(audio_np)
                    audio = audio.unsqueeze(0) if audio.dim() == 1 else audio.t()
                    if sr != 44100: continue
                    if audio.shape[0] == 1: audio = audio.repeat(2, 1)
                    gt_audios.append(audio)

                if len(gt_audios) < num_stems: continue

                min_len = min(a.shape[1] for a in gt_audios)
                gt_audios = [a[:, :min_len].to(device) for a in gt_audios]
                mixture = torch.stack(gt_audios).sum(dim=0)

                with torch.no_grad():
                    pred_audios = inference(model, None, mixture, chunk_size, overlap, device, return_tensors=True)

                stems_sdr = [calculate_sdr(p, g) for p, g in zip(pred_audios, gt_audios)]
                for i, s in enumerate(stems_sdr):
                    total_stems_sdr[i] += s
                
                track_sdr = sum(stems_sdr) / num_stems
                total_sdr += track_sdr
                count += 1
                sdr_info = " | ".join([f"{STEMS[i].capitalize()}: {stems_sdr[i]:.4f} SDR" for i in range(num_stems)])
                pbar.set_postfix_str(sdr_info)
            except Exception:
                continue

    if count == 0:
        print("No valid test files processed.")
        return [0.0] * num_stems, 0.0

    avg_stems_sdr = [s / count for s in total_stems_sdr]
    return avg_stems_sdr, total_sdr / count

def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "").replace("._orig_mod", "")
        cleaned[new_k] = v
    return cleaned

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {self.clean_name(name): param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def clean_name(self, name):
        return name.replace("_orig_mod.", "").replace("._orig_mod", "")

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    clean = self.clean_name(name)
                    self.shadow[clean].copy_(self.decay * self.shadow[clean] + (1.0 - self.decay) * param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                clean = self.clean_name(name)
                self.backup[clean] = param.data.clone()
                param.data.copy_(self.shadow[clean])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                clean = self.clean_name(name)
                param.data.copy_(self.backup[clean])
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        for name, value in state_dict.items():
            clean = self.clean_name(name)
            if clean in self.shadow:
                self.shadow[clean].copy_(value)

def train(model, dataloader, optimizer, loss_fn, device, checkpoint_steps, args, segment_length, checkpoint_path=None, window=None, reset_optimizer=False):
    model.to(device)
    ema = EMA(model, decay=0.999)
    step = 0
    avg_loss = 0.0
    scaler = GradScaler()

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
        model.load_state_dict(clean_state_dict(checkpoint_data['model_state_dict']), strict=False)

        if 'ema_state_dict' in checkpoint_data:
            ema.load_state_dict(clean_state_dict(checkpoint_data['ema_state_dict']))
            print("EMA state loaded from checkpoint.")
        else:
            print("No EMA state found in checkpoint. Initializing from current model weights.")
        
        step = checkpoint_data.get('step', 0)
        avg_loss = checkpoint_data.get('avg_loss', 0.0)

        if not reset_optimizer and 'optimizer_state_dict' in checkpoint_data:
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            for i, group in enumerate(optimizer.param_groups):
                group['lr'] = current_lrs[i]
            print(f"Resuming training from step {step}. MODEL AND OPTIMIZER LOADED.")
        else:
            print(f"Resuming training from step {step}. MODEL LOADED, OPTIMIZER RESET.")

    current_lrs = [group['lr'] for group in optimizer.param_groups]
    if len(set(current_lrs)) == 1:
        print(f"Currently applied learning rate: {current_lrs[0]}")
    else:
        print(f"Currently applied learning rates: {current_lrs}")

    progress_bar = tqdm(initial=step, total=None, dynamic_ncols=True)

    if args.ckpt:
        for i in range(len(model.model.time_layers)):
            model.model.time_layers[i] = torch.compile(model.model.time_layers[i])
            model.model.freq_layers[i] = torch.compile(model.model.freq_layers[i])
        compiled_model = model
    else:
        compiled_model = torch.compile(model)

    while True:
        compiled_model.train()
        for batch in dataloader:
            mixture_spec, target_audios = batch

            mixture_spec = mixture_spec.to(device, non_blocking=True)
            target_audios = target_audios.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                pred_masks = compiled_model(mixture_spec)
                loss = loss_fn(pred_masks, mixture_spec, target_audios, stft_params_for_istft, multi_res_complex_loss_calculator)

            if torch.isnan(loss).any(): continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            avg_loss = 0.999 * avg_loss + 0.001 * loss.item() if step > 0 else loss.item()
            step += 1
            progress_bar.update(1)
            progress_bar.set_description(f"Step {step} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f} - Best SDR: {best_sdr:.4f}")

            if step > 0 and step % checkpoint_steps == 0:
                checkpoint_payload = {
                    'step': step, 
                    'model_state_dict': clean_state_dict(model.state_dict()), 
                    'ema_state_dict': clean_state_dict(ema.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'stems': STEMS
                }

                reg_checkpoint_path = f"ckpts/checkpoint_step_{step}.pt"
                torch.save(checkpoint_payload, reg_checkpoint_path)

                regular_checkpoints = sorted(
                    [os.path.join('ckpts', f) for f in os.listdir('ckpts') if f.endswith('.pt')],
                    key=os.path.getmtime
                )
                if len(regular_checkpoints) > 3:
                    os.remove(regular_checkpoints[0])

                ema.apply_shadow()
                avg_stems_sdr, avg_combined_sdr = validate(model, args.test_dir, device, segment_length, overlap=88200)
                ema.restore()

                sdr_str = ", ".join([f"{STEMS[i].capitalize()} SDR: {avg_stems_sdr[i]:.4f}" for i in range(len(STEMS))])
                print(f"\nValidation Step {step} (EMA): {sdr_str}, Combined SDR: {avg_combined_sdr:.4f}")

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
                
                compiled_model.train()

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

def inference(model, checkpoint_path, input_data,
              chunk_size=485100, overlap=88200, device='cpu', return_tensors=False):
    global STEMS
    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'stems' in checkpoint_data:
            STEMS = checkpoint_data['stems']
        model.load_state_dict(clean_state_dict(checkpoint_data['model_state_dict']), strict=False)

    num_stems = len(STEMS)
    model.eval().to(device)

    if isinstance(input_data, str):
        input_audio_np, sr = sf.read(input_data, dtype='float32')
        input_audio = torch.from_numpy(input_audio_np)
        input_audio = input_audio.unsqueeze(0) if input_audio.dim() == 1 else input_audio.t()
        if sr != 44100: raise ValueError(f"Input audio must be 44100Hz, but got {sr}Hz.")
    else:
        input_audio = input_data

    if input_audio.shape[0] == 1: input_audio = input_audio.repeat(2, 1)
    elif input_audio.shape[0] != 2: raise ValueError("Input audio must be mono or stereo.")
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

    with tqdm(total=num_chunks, desc="Processing audio", leave=False) as pbar:
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

            with torch.no_grad():
                with autocast():
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
                stem_chunk_full = torch.istft(stem_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=target_length, center=True)
                pred_stems[j][:, start:actual_end] += stem_chunk_full[:, :usable] * w[:usable]
            
            sum_weights[start:actual_end] += w[:usable]
            pbar.update(1)

    sum_weights = sum_weights.clamp(min=1e-8)
    for j in range(num_stems):
        pred_stems[j] = (pred_stems[j] / sum_weights).clamp(-1.0, 1.0)

    if return_tensors:
        return pred_stems
    else:
        os.makedirs('outputs', exist_ok=True)
        for j, stem_name in enumerate(STEMS):
            path = os.path.join('outputs', f"{stem_name}.wav")
            print(f"Saving {stem_name} to {path}...")
            sf.write(path, pred_stems[j].cpu().numpy().T, 44100)
        print("Done.")

def main():
    if len(STEMS) < 2:
        print("Error: At least 2 stems must be defined in STEMS.")
        return

    parser = argparse.ArgumentParser(description='Train or run inference on a source separation model.')
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--infer', action='store_true', help='Run inference.')
    parser.add_argument('--data_dir', type=str, default='train', help='Path to the training dataset.')
    parser.add_argument('--test_dir', type=str, default='test', help='Path to the test dataset for validation.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--checkpoint_steps', type=int, default=4000, help='Save a checkpoint every X steps.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Specific checkpoint path to resume training or for inference. Overrides automatic selection.')
    parser.add_argument('--input_file', type=str, default=None, help='Path to the input audio file for inference.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers.')
    parser.add_argument('--reset_optimizer', action='store_true', help='Reset optimizer state when resuming from a checkpoint.')
    parser.add_argument('--latest', action='store_true', help='Use the latest checkpoint for inference instead of the best SDR one.')
    parser.add_argument('--ckpt', action='store_true', help='Enable gradient checkpointing to reduce memory usage during training.')

    args = parser.parse_args()

    segment_length = 264600

    os.makedirs('ckpts', exist_ok=True)
    os.makedirs('best_ckpts', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel(use_checkpoint=args.ckpt)
    optimizer = AdamAtan2(model.parameters(), lr=1e-5)

    if args.train:
        checkpoint_to_load = args.checkpoint_path
        if not checkpoint_to_load:
            checkpoint_to_load = find_latest_checkpoint('ckpts')
            if checkpoint_to_load:
                print(f"Automatically resuming from latest checkpoint: {checkpoint_to_load}")

        train_dataset = Dataset(root_dir=args.data_dir, segment_length=segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        train(model, train_dataloader, optimizer, loss_fn, device, args.checkpoint_steps, args, segment_length, checkpoint_path=checkpoint_to_load, window=window, reset_optimizer=args.reset_optimizer)

    elif args.infer:
        if not args.input_file:
            print("Error: --input_file is required for inference.")
            return

        checkpoint_to_load = args.checkpoint_path
        if not checkpoint_to_load:
            checkpoint_to_load = find_latest_checkpoint('ckpts') if args.latest else find_best_sdr_checkpoint('best_ckpts')
            if checkpoint_to_load:
                print(f"No checkpoint specified. Automatically using {'latest' if args.latest else 'best SDR'} checkpoint: {checkpoint_to_load}")
        
        if not checkpoint_to_load:
            print("Error: No checkpoint found.")
            return

        inference(model, checkpoint_to_load, args.input_file,
                  chunk_size=segment_length, device=device)

    else:
        print("Please specify either --train or --infer.")

if __name__ == '__main__':
    main()