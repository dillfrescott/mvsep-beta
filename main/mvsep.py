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
import numpy as np
import re
from encoder import Encoder
import warnings

warnings.filterwarnings("ignore")

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049,
                 embed_dim=256, depth=12, heads=8, n_bands=60):
        super().__init__()
        self.freq_bins = freq_bins
        self.n_bands = n_bands
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        def get_band_widths(n_bins, n_bands):
            splits = np.logspace(np.log10(1), np.log10(n_bins), n_bands + 1)
            splits = np.unique(np.round(splits).astype(int))
            if splits[0] == 0: splits[0] = 1
            if splits[-1] != n_bins: splits[-1] = n_bins
            
            widths = np.diff(splits).tolist()
            
            while len(widths) < n_bands:
                widths.insert(0, 1)
            
            widths[-1] += (n_bins - sum(widths))
                
            return widths

        self.band_widths = get_band_widths(freq_bins, n_bands)
        self.max_band_width = max(self.band_widths)
        
        indices = []
        for i, w in enumerate(self.band_widths):
            indices.extend([i * self.max_band_width + j for j in range(w)])
        self.register_buffer('freq_index_map', torch.tensor(indices, dtype=torch.long))

        self.out_factor = self.out_masks * 2
        self.max_out_channels = self.max_band_width * self.out_factor

        self.band_proj1 = nn.Conv2d(
            in_channels * 2 * self.n_bands, 
            embed_dim * self.n_bands, 
            kernel_size=(self.max_band_width, 1), 
            groups=self.n_bands
        )
        self.act = nn.GELU()
        self.band_proj2 = nn.Conv2d(
            embed_dim * self.n_bands, 
            embed_dim * self.n_bands, 
            kernel_size=(1, 3), 
            padding=(0, 1), 
            groups=self.n_bands
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.time_encoder = Encoder(
            dim=embed_dim,
            depth=depth,
            heads=heads
        )

        self.freq_mixer = Encoder(
            dim=embed_dim,
            depth=2,
            heads=heads
        )

        self.output_proj1 = nn.Conv2d(
            embed_dim * self.n_bands, 
            embed_dim * self.n_bands, 
            kernel_size=(1, 3), 
            padding=(0, 1), 
            groups=self.n_bands
        )
        self.output_proj2 = nn.Conv2d(
            embed_dim * self.n_bands, 
            self.max_out_channels * self.n_bands, 
            kernel_size=(1, 1), 
            groups=self.n_bands
        )

    def forward(self, x_stft, x_audio):
        B, _, _, T = x_stft.shape
        
        x_stft = torch.cat([x_stft.real, x_stft.imag], dim=1)

        bands = torch.split(x_stft, self.band_widths, dim=2)
        bands_padded = [
            F.pad(b, (0, 0, 0, self.max_band_width - b.shape[2])) 
            for b in bands
        ]
        x = torch.cat(bands_padded, dim=1)

        x = self.band_proj1(x)
        x = self.act(x)
        x = self.band_proj2(x)

        x = x.view(B, self.n_bands, self.embed_dim, 1, T)
        x = x.squeeze(3).permute(0, 1, 3, 2).contiguous()
        
        x = self.norm(x)
        x = x.view(B * self.n_bands, T, self.embed_dim)
        x = self.time_encoder(x)
        x = x.view(B, self.n_bands, T, self.embed_dim)

        x = x.permute(0, 2, 1, 3).contiguous().view(B * T, self.n_bands, self.embed_dim)
        x = self.freq_mixer(x)
        x = x.view(B, T, self.n_bands, self.embed_dim)

        x = x.permute(0, 2, 3, 1).contiguous().view(B, self.n_bands * self.embed_dim, 1, T)
        
        x = self.output_proj1(x)
        x = self.act(x)
        x = self.output_proj2(x)

        x = x.view(B, self.n_bands, self.max_out_channels, T)
        x = x.view(B, self.n_bands, self.out_factor, self.max_band_width, T)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.out_factor, self.n_bands * self.max_band_width, T)
        
        x = x[:, :, self.freq_index_map, :]

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

            real_loss = F.l1_loss(stft_pred.real, stft_true.real)
            imag_loss = F.l1_loss(stft_pred.imag, stft_true.imag)

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
    
    total_loss = vocal_loss + instr_loss

    return total_loss

class Dataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_length=88200, segment=True, noise_level=0.0):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment = segment
        self.noise_level = noise_level

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

                if self.noise_level > 0:
                    noise_factor = self.noise_level * (1 + (torch.rand(1).item() * 2 - 1) * 0.5)
                    vocal_noise = torch.randn_like(vocal_seg) * noise_factor
                    instr_noise = torch.randn_like(instr_seg) * noise_factor
                    vocal_seg = vocal_seg + vocal_noise
                    instr_seg = instr_seg + instr_noise

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

def train(model, dataloader, optimizer, loss_fn, device, checkpoint_steps, args, segment_length, checkpoint_path=None, window=None, reset_optimizer=False):
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

                avg_vocal_sdr, avg_instr_sdr, avg_combined_sdr = validate(model, args.test_dir, device, segment_length, overlap=88200)
                
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
            chunk = input_audio[:, start:end]
            L = chunk.shape[1]

            if L < n_fft:
                pbar.update(1)
                continue

            target_length = chunk_size
            padded_chunk = chunk
            if L < target_length:
                padding_amount = target_length - L
                padded_chunk = F.pad(chunk, (0, padding_amount), 'constant', 0)

            spec = torch.stft(padded_chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True)

            with torch.no_grad():
                pred_output = model(spec.unsqueeze(0), padded_chunk.unsqueeze(0)).squeeze(0)

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

            vocal_chunk_padded = torch.istft(vocal_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=target_length, center=True)
            inst_chunk_padded = torch.istft(instrumental_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=target_length, center=True)

            vocal_chunk = vocal_chunk_padded[:, :L]
            inst_chunk = inst_chunk_padded[:, :L]

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
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--checkpoint_steps', type=int, default=4000, help='Save a checkpoint every X steps.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Specific checkpoint path to resume training or for inference. Overrides automatic selection.')
    parser.add_argument('--input_file', type=str, default=None, help='Path to the input audio file for inference.')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path for the output instrumental file.')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path for the output vocal file.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers.')
    parser.add_argument('--reset_optimizer', action='store_true', help='Reset optimizer state when resuming from a checkpoint.')
    parser.add_argument('--latest', action='store_true', help='Use the latest checkpoint for inference instead of the best SDR one.')

    args = parser.parse_args()

    noise_level = 0.005
    segment_length = 441000

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

        train_dataset = Dataset(root_dir=args.data_dir, segment_length=segment_length, segment=True, noise_level=noise_level)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        train(model, train_dataloader, optimizer, loss_fn, device, args.checkpoint_steps, args, segment_length, checkpoint_path=checkpoint_to_load, window=window, reset_optimizer=args.reset_optimizer)
    
    elif args.infer:
        if not args.input_file:
            print("Error: --input_file is required for inference.")
            return

        checkpoint_to_load = args.checkpoint_path
        if not checkpoint_to_load:
            if args.latest:
                checkpoint_to_load = find_latest_checkpoint('ckpts')
                if checkpoint_to_load:
                    print(f"No checkpoint specified. Automatically using latest checkpoint: {checkpoint_to_load}")
                else:
                    print("Error: --latest specified but no checkpoints found in 'ckpts/'.")
                    return
            else:
                checkpoint_to_load = find_best_sdr_checkpoint('best_ckpts')
                if checkpoint_to_load:
                    print(f"No checkpoint specified. Automatically using best SDR checkpoint: {checkpoint_to_load}")
                else:
                    print("Error: No checkpoint specified with --checkpoint_path and no best SDR checkpoint was found in 'best_ckpts/'.")
                    return
        else:
            print(f"Using specified checkpoint: {checkpoint_to_load}")

        inference(model, checkpoint_to_load, args.input_file, args.output_instrumental, args.output_vocal,
                  chunk_size=segment_length, device=device)
    
    else:
        print("Please specify either --train or --infer.")

if __name__ == '__main__':
    main()
