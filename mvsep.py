import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import auraloss
import numpy as np
import random
import math
import glob
from torch.utils.checkpoint import checkpoint

def apply_rotary_emb(x, freqs):
    batch, channels, freq, time = x.shape

    # Permute to [batch, freq, time, channels] and flatten spatial dims.
    x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, channels)

    # Make sure freqs is broadcastable over batch, freq, time.
    if freqs.shape[1] != channels:
        repeat_times = math.ceil(channels / freqs.shape[1])
        freqs = freqs.repeat(1, repeat_times, 1, 1)[:, :channels, :, :]
    freqs = freqs.reshape(-1, channels)

    # Ensure channels is even for complex representation.
    if channels % 2 != 0:
        channels_adjusted = channels - 1
        x_reshaped = x_reshaped[:, :channels_adjusted]
        freqs = freqs[:, :channels_adjusted]
    else:
        channels_adjusted = channels

    # Reshape into pairs for complex arithmetic.
    x_pairs = x_reshaped.float().reshape(-1, channels_adjusted // 2, 2)
    freqs_pairs = freqs.float().reshape(-1, channels_adjusted // 2, 2)

    # Convert pairs to complex numbers and apply rotation.
    x_complex = torch.view_as_complex(x_pairs.contiguous())
    freqs_complex = torch.view_as_complex(freqs_pairs.contiguous())
    x_rotated = x_complex * torch.exp(1j * freqs_complex)

    # Convert back to real and flatten the pairs.
    x_rotated = torch.view_as_real(x_rotated).flatten(1)
    
    # If channels were odd, append the unrotated last channel.
    if channels % 2 != 0:
        x_remaining = x[:, -1:, :, :].reshape(batch * freq * time, 1)
        x_rotated = torch.cat([x_rotated, x_remaining.float()], dim=-1)

    # Reshape back to [batch, channels, freq, time].
    x_rotated = x_rotated.reshape(batch, freq, time, channels).permute(0, 3, 1, 2)
    return x_rotated

class DARPE(nn.Module):
    def __init__(self, dim, in_channels=None, max_freq=10000, init_scale=1.0, mlp_hidden=256):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.base_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        
        # If in_channels is not provided, default to using 'dim'
        in_channels = in_channels if in_channels is not None else dim
        
        # Register projection if needed.
        if in_channels != dim:
            self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
            
        self.adaptive_mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, dim, kernel_size=1),
            nn.Tanh()
        )
    
    def _get_1d_freqs(self, pos, scale):
        half_dim = self.dim // 2
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=pos.device)
        # Compute frequencies as in standard sinusoidal PE.
        dim_t = self.max_freq ** (-2 * dim_t / half_dim)
        pos = pos.unsqueeze(-1) / scale
        freqs = pos * dim_t.unsqueeze(0)
        return freqs

    def get_freqs(self, F_dim, T_dim):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create positional indices for frequency and time axes.
        h_pos = torch.arange(F_dim, device=device).float()
        t_pos = torch.arange(T_dim, device=device).float()
        # Get 1D frequencies for each axis.
        freqs_h = self._get_1d_freqs(h_pos, F_dim)
        freqs_t = self._get_1d_freqs(t_pos, T_dim)
        # Create a 2D grid of frequencies.
        freqs = torch.zeros(F_dim, T_dim, self.dim, device=device)
        freqs[..., 0::2] = freqs_h.unsqueeze(1).expand(-1, T_dim, -1)
        freqs[..., 1::2] = freqs_t.unsqueeze(0).expand(F_dim, -1, -1)
        return freqs  # Shape: [F_dim, T_dim, dim]
    
    def forward(self, x):
        B, channels, F_dim, T_dim = x.shape
            
        # Compute frequencies.
        freqs = self.get_freqs(F_dim, T_dim)  # shape [F_dim, T_dim, dim]
        freqs = freqs * self.base_scale
        freqs = freqs.permute(2, 0, 1).unsqueeze(0)  # [1, dim, F, T]
        
        if channels > self.dim:
            repeat_times = math.ceil(channels / self.dim)
            freqs = freqs.repeat(1, repeat_times, 1, 1)[:, :channels, :, :]
        
        # Use the registered projection.
        x_proj = self.proj(x)
        adaptive = self.adaptive_mlp(x_proj)  # [B, dim, F, T]
        
        # Clamp the adaptive scaling factor to prevent extreme values
        adaptive = torch.clamp(adaptive, min=-1.0, max=1.0)
        
        freqs = freqs.expand(B, -1, -1, -1)
        adapted_freqs = freqs * (1 + adaptive)
        
        # Safe complex operations
        x_rotated = apply_rotary_emb(x, adapted_freqs)
        
        # Add a small epsilon to prevent NaN in gradients
        x_rotated = x_rotated + 1e-8
        return x_rotated

class DualMaskPredictor(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        # Output two channels: one for vocals and one for instrumentals.
        self.out_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
    
    def forward(self, x):
        # x is expected to have shape [B, hidden_channels, F, T].
        x = self.conv_block(x)
        logits = self.out_conv(x)  # shape: [B, 2, F, T]
        # Softmax produces a probability distribution between the two classes per bin.
        masks = torch.softmax(logits, dim=1)
        # Extract vocal mask
        vocal_mask = masks[:, 0:1, :, :]
        return vocal_mask

def apply_gru(gru_layer, x):
    batch_size, channels, freq, time = x.shape
    x_gru_input = x.permute(0, 2, 3, 1).reshape(batch_size * freq, time, channels)
    x_gru_output, _ = gru_layer(x_gru_input)
    output_channels = gru_layer.hidden_size * (2 if gru_layer.bidirectional else 1)
    x_out = x_gru_output.view(batch_size, freq, time, output_channels).permute(0, 3, 1, 2)
    return x_out

class EncoderGRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gru_layers=1, bidirectional=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        gru_hidden_size = out_channels // 2 if bidirectional else out_channels
        self.gru = nn.GRU(
            input_size=out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.gru_out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x_gru = apply_gru(self.gru, x)
        skip = x_gru

        F_dim = x_gru.shape[2]
        T_dim = x_gru.shape[3]

        pool_kernel_F = 2 if F_dim > 1 else 1
        pool_kernel_T = 2 if T_dim > 1 else 1

        if pool_kernel_F > 1 or pool_kernel_T > 1:
            out = F.max_pool2d(x_gru, kernel_size=(pool_kernel_F, pool_kernel_T))
        else:
            out = x_gru

        return out, skip

class DecoderGRUBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, gru_layers=1, bidirectional=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        combined_channels = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        gru_hidden_size = out_channels // 2 if bidirectional else out_channels
        self.gru = nn.GRU(
            input_size=out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.gru_out_channels = out_channels

    def forward(self, x, skip):
        x = self.upsample(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x_out = apply_gru(self.gru, x)
        return x_out

class BottleneckGRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gru_layers=1, bidirectional=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        gru_hidden_size = out_channels // 2 if bidirectional else out_channels
        self.gru = nn.GRU(
            input_size=out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.gru_out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x_out = apply_gru(self.gru, x)
        return x_out

class GRUUWNet(nn.Module):
    def __init__(self, in_channels, base_hidden_channels, depth=3, gru_layers=1, bidirectional=False):
        super().__init__()
        
        self.encoders1 = nn.ModuleList()
        self.decoders1 = nn.ModuleList()
        self.encoders2 = nn.ModuleList()
        self.decoders2 = nn.ModuleList()
        
        ch1 = base_hidden_channels
        current_in_channels1 = in_channels
        for i in range(depth):
            out_ch = ch1 * 2
            self.encoders1.append(EncoderGRUBlock(current_in_channels1, out_ch, gru_layers, bidirectional))
            current_in_channels1 = out_ch
            ch1 *= 2
            
        self.bottleneck1 = BottleneckGRUBlock(ch1, ch1 * 2, gru_layers, bidirectional)
        ch1 *= 2
        
        for i in range(depth):
            skip_ch = ch1 // 2 
            out_ch = ch1 // 2
            self.decoders1.append(DecoderGRUBlock(ch1, skip_ch, out_ch, gru_layers, bidirectional))
            ch1 //= 2
            
        first_wnet_out_channels = base_hidden_channels
        self.final_conv1 = nn.Conv2d(ch1, first_wnet_out_channels, kernel_size=1) 

        ch2 = base_hidden_channels
        current_in_channels2 = first_wnet_out_channels 
        for i in range(depth):
            out_ch = ch2 * 2
            self.encoders2.append(EncoderGRUBlock(current_in_channels2, out_ch, gru_layers, bidirectional))
            current_in_channels2 = out_ch
            ch2 *= 2
            
        self.bottleneck2 = BottleneckGRUBlock(ch2, ch2 * 2, gru_layers, bidirectional)
        ch2 *= 2
        
        for i in range(depth):
            skip_ch = ch2 // 2
            out_ch = ch2 // 2
            self.decoders2.append(DecoderGRUBlock(ch2, skip_ch, out_ch, gru_layers, bidirectional))
            ch2 //= 2
            
        self.final_conv2 = nn.Conv2d(ch2, base_hidden_channels, kernel_size=1)


    def forward(self, x):
        skips1 = []
        x1 = x
        for encoder in self.encoders1:
            x1, skip = encoder(x1)
            skips1.append(skip)
        
        x1 = self.bottleneck1(x1)
        
        skips1 = skips1[::-1]
        for i, decoder in enumerate(self.decoders1):
            x1 = decoder(x1, skips1[i])
            
        x1_out = self.final_conv1(x1)

        skips2 = []
        x2 = x1_out
        for encoder in self.encoders2:
            x2, skip = encoder(x2)
            skips2.append(skip)
            
        x2 = self.bottleneck2(x2)
        
        skips2 = skips2[::-1]
        for i, decoder in enumerate(self.decoders2):
            x2 = decoder(x2, skips2[i])
            
        x2_out = self.final_conv2(x2)
        
        return x2_out

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=84, num_wnet_layers=2, gru_layers_per_block=1, bidirectional_gru=False):
        super(NeuralModel, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        )
        self.darpe = DARPE(dim=hidden_channels) 
        self.gru_w_net = GRUUWNet(
            in_channels=hidden_channels, 
            base_hidden_channels=hidden_channels,
            depth=num_wnet_layers,
            gru_layers=gru_layers_per_block,
            bidirectional=bidirectional_gru
        )
        self.mask_predictor = DualMaskPredictor(hidden_channels)

    def forward(self, x):
        x = self.projection(x)
        x = self.darpe(x)
        x = self.gru_w_net(x) 
        vocal_mask = self.mask_predictor(x)
        final_vocal_mask = vocal_mask.expand(-1, 2, -1, -1) 
        return final_vocal_mask

def loss_fn(pred_vocal_mask,
            target_vocal_mag,
            target_instrumental_mag,
            mixture_mag,
            mixture_phase,
            window, n_fft, hop_length):

    device = target_vocal_mag.device
    stft_loss_calculator = auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        perceptual_weighting=True,
        sample_rate=44100,
        scale="mel",
        n_bins=128,
        device=device
    )

    pred_vocal_mask = torch.clamp(pred_vocal_mask, 0.0, 1.0)

    # Apply the mask to the mixture magnitude to get predicted source magnitudes
    pred_vocal_mag = mixture_mag * pred_vocal_mask
    # The instrumental mask is implicitly (1 - vocal_mask)
    pred_instrumental_mag = mixture_mag * (1.0 - pred_vocal_mask)

    # Clamp all magnitudes (predicted and target) to avoid log(0) or division by zero issues
    pred_vocal_mag = torch.clamp(pred_vocal_mag, min=1e-8)
    pred_instrumental_mag = torch.clamp(pred_instrumental_mag, min=1e-8)
    target_vocal_mag = torch.clamp(target_vocal_mag, min=1e-8)
    target_instrumental_mag = torch.clamp(target_instrumental_mag, min=1e-8)

    # L1 loss on magnitudes (now comparing derived pred_mags with targets)
    l1_vocal_loss = F.l1_loss(pred_vocal_mag, target_vocal_mag)
    l1_instrumental_loss = F.l1_loss(pred_instrumental_mag, target_instrumental_mag)

    def make_complex(mag, phase):
        # Ensure phase has the same shape as mag if necessary (it should already match)
        return torch.polar(mag, phase)

    def istft_channels(spec, n_fft, hop_length, window):
        batch_size, channels, F_dim, T_dim = spec.shape
        spec_combined = spec.reshape(batch_size * channels, F_dim, T_dim)
        # Ensure window is on the correct device
        window = window.to(spec_combined.device)
        audio = torch.istft(
            spec_combined,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window.shape[0],
            window=window,
            return_complex=False,
            center=True
        )
        audio = audio.reshape(batch_size, channels, -1)
        return audio

    # Use mixture phase for consistency
    pred_vocal_spec = make_complex(pred_vocal_mag, mixture_phase)
    target_vocal_spec = make_complex(target_vocal_mag, mixture_phase)
    pred_instrumental_spec = make_complex(pred_instrumental_mag, mixture_phase)
    target_instrumental_spec = make_complex(target_instrumental_mag, mixture_phase)

    pred_vocal_audio = istft_channels(pred_vocal_spec, n_fft, hop_length, window)
    target_vocal_audio = istft_channels(target_vocal_spec, n_fft, hop_length, window)
    pred_instrumental_audio = istft_channels(pred_instrumental_spec, n_fft, hop_length, window)
    target_instrumental_audio = istft_channels(target_instrumental_spec, n_fft, hop_length, window)

    # Trimming to same length
    min_len = min(pred_vocal_audio.shape[-1], target_vocal_audio.shape[-1],
                  pred_instrumental_audio.shape[-1], target_instrumental_audio.shape[-1])
    pred_vocal_audio = pred_vocal_audio[..., :min_len]
    target_vocal_audio = target_vocal_audio[..., :min_len]
    pred_instrumental_audio = pred_instrumental_audio[..., :min_len]
    target_instrumental_audio = target_instrumental_audio[..., :min_len]

    # Calculate STFT-based reconstruction losses
    vocal_reconstruction_loss = stft_loss_calculator(pred_vocal_audio, target_vocal_audio)
    instrumental_reconstruction_loss = stft_loss_calculator(pred_instrumental_audio, target_instrumental_audio)

    # Penalizes similarity between predicted vocals and target instrumentals (and vice versa)
    dissimilarity_v = F.l1_loss(pred_vocal_audio, target_instrumental_audio)
    dissimilarity_i = F.l1_loss(pred_instrumental_audio, target_vocal_audio)

    total_loss = (
        l1_vocal_loss +
        l1_instrumental_loss +
        vocal_reconstruction_loss +
        instrumental_reconstruction_loss -
        (dissimilarity_v + dissimilarity_i)
    )

    return total_loss

class MUSDBDataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_length=88200, segment=True):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment = segment

        # STFT parameters
        self.n_fft = 4096
        self.hop_length = 1024
        self.window = torch.hann_window(self.n_fft)

        # Gather lists of available vocal and instrumental stems.
        self.track_dirs = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]
        self.vocal_paths = []   # List of file paths for vocals.
        self.instr_paths = []   # List of tuples for instrumental components (drums, bass, other).

        print("Scanning track folders for stems...")
        for td in tqdm(self.track_dirs, desc="Scanning tracks"):
            vocal_path = os.path.join(td, 'vocals.wav')
            drum_path = os.path.join(td, 'drums.wav')
            bass_path = os.path.join(td, 'bass.wav')
            other_path = os.path.join(td, 'other.wav')
            
            if os.path.exists(vocal_path):
                self.vocal_paths.append(vocal_path)
            # For instrumentals, require all three components to be present.
            if os.path.exists(drum_path) and os.path.exists(bass_path) and os.path.exists(other_path):
                self.instr_paths.append((drum_path, bass_path, other_path))
                
        # Ensure we have at least some pairs.
        if not self.vocal_paths or not self.instr_paths:
            raise ValueError("Dataset must contain both vocal and instrumental stems.")
        
        # For random pairing during training, set an arbitrary large dataset size.
        self.size = 50000

    def _preprocess_audio(self, audio, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        # Ensure stereo: if mono then duplicate; if more than 2 channels then take first two.
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        return audio

    def _load_audio(self, filepath):
        audio, sr = torchaudio.load(filepath)
        audio = self._preprocess_audio(audio, sr)
        return audio

    def _load_vocal(self, path):
        return self._load_audio(path)

    def _load_instrumental(self, paths):
        audios = []
        min_length = float("inf")
        for p in paths:
            audio = self._load_audio(p)
            audios.append(audio)
            min_length = min(min_length, audio.shape[1])
        # Truncate and sum the components
        summed_audio = sum([audio[:, :min_length] for audio in audios])
        return summed_audio

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Randomly choose a vocal track and an instrumental track (they may be from different songs)
        vocal_path = random.choice(self.vocal_paths)
        instr_tuple = random.choice(self.instr_paths)
        
        # Load raw audio
        vocal_audio = self._load_vocal(vocal_path)      # Shape: [2, num_samples]
        instr_audio = self._load_instrumental(instr_tuple)  # Shape: [2, num_samples]
        
        # Match lengths: use the minimum length available.
        min_length = min(vocal_audio.shape[1], instr_audio.shape[1])
        if min_length == 0:
            raise ValueError("Encountered an audio file with zero length.")
        
        # Select a segment if needed.
        if self.segment and self.segment_length < min_length:
            start = random.randint(0, min_length - self.segment_length)
            end = start + self.segment_length
        else:
            start = 0
            end = min_length
        
        vocal_seg = vocal_audio[:, start:end]
        instr_seg = instr_audio[:, start:end]

        # Compute STFTs for each segment.
        vocal_spec = torch.stft(vocal_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window, return_complex=True)
        instr_spec = torch.stft(instr_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window, return_complex=True)
        
        # Compute magnitudes.
        vocal_mag = torch.abs(vocal_spec)
        instr_mag = torch.abs(instr_spec)
        
        # Create the mixture: sum the complex spectrograms.
        mixture_spec = vocal_spec + instr_spec
        mixture_mag = torch.abs(mixture_spec)
        mixture_phase = torch.angle(mixture_spec)
        
        return mixture_mag, mixture_phase, vocal_mag, instr_mag

def adjust_learning_rate(optimizer, grad_norm, base_lr, scale=1.0, eps=1e-8):
    grad_norm = max(grad_norm, eps)
    lr = base_lr * (1.0 / (1.0 + grad_norm / scale))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, dataloader, optimizer, loss_fn, device, epochs, checkpoint_steps, args, checkpoint_path=None, window=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    checkpoint_files = []

    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        step = checkpoint_data['step']
        avg_loss = checkpoint_data['avg_loss']
        print(f"Resuming training from step {step} with average loss {avg_loss:.4f}")

    progress_bar = tqdm(total=epochs * len(dataloader))
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            mixture_mag, mixture_phase, vocal_mag, instrumental_mag = batch
            mixture_mag = mixture_mag.to(device)
            mixture_phase = mixture_phase.to(device)
            vocal_mag = vocal_mag.to(device)
            instrumental_mag = instrumental_mag.to(device)

            optimizer.zero_grad()
            pred_vocal_mask = model(mixture_mag)
            loss = loss_fn(pred_vocal_mask, vocal_mag, instrumental_mag, mixture_mag, mixture_phase, window, 4096, 1024)
            
            if torch.isnan(loss).any():
                print("NaN loss detected, skipping batch")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            adjust_learning_rate(optimizer, grad_norm, base_lr=args.learning_rate)
            optimizer.step()

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            step += 1
            progress_bar.update(1)
            current_lr = optimizer.param_groups[0]['lr']
            desc = f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.8f}"
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
    progress_bar.close()

def inference(model, checkpoint_path, input_wav_path, output_instrumental_path, output_vocal_path,
              chunk_size=88200, overlap=44100, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval()
    model.to(device)
    
    # Load audio
    input_audio, sr = torchaudio.load(input_wav_path)
    if sr != 44100:
        raise ValueError(f"Input audio must be 44100Hz, but got {sr}Hz. Please resample the audio first.")
    if input_audio.shape[0] != 2:
        if input_audio.shape[0] == 1:
            input_audio = input_audio.repeat(2, 1)
        else:
            raise ValueError("Input audio must be mono or stereo")
        
    input_audio = input_audio.to(device)
    total_length = input_audio.shape[1]
    vocals = torch.zeros_like(input_audio)
    instrumentals = torch.zeros_like(input_audio)
    cross_fade_length = overlap // 2
    window = torch.hann_window(4096).to(device)
    n_fft = 4096
    hop_length = 1024
    min_chunk_size = n_fft  # Minimum size needed for STFT
    
    # Calculate number of chunks
    step_size = max(1, chunk_size - overlap)
    num_chunks = math.ceil(max(0, total_length - overlap) / step_size)
    
    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length, step_size):
            end = min(i + chunk_size, total_length)
            chunk = input_audio[:, i:end]
            chunk_length = chunk.shape[1]
            
            # Skip chunks that are too small
            if chunk_length < min_chunk_size:
                if i == 0:
                    # Pad first chunk if too small
                    pad_amount = min_chunk_size - chunk_length
                    chunk = F.pad(chunk, (0, pad_amount))
                    chunk_length = chunk.shape[1]
                else:
                    # For other small chunks, just skip them
                    pbar.update(1)
                    continue
            
            # Process the chunk
            try:
                chunk_spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, 
                                      window=window, return_complex=True)
            except RuntimeError as e:
                print(f"Skipping chunk at position {i}-{end} due to error: {str(e)}")
                pbar.update(1)
                continue
            
            chunk_mag = torch.abs(chunk_spec)
            chunk_phase = torch.angle(chunk_spec)
            
            with torch.no_grad():
                pred_vocal_mask = model(chunk_mag.unsqueeze(0)).squeeze(0)
            
            pred_vocal_mag = chunk_mag * pred_vocal_mask
            pred_instrumental_mag = chunk_mag * (1 - pred_vocal_mask)

            vocal_spec = pred_vocal_mag * torch.exp(1j * chunk_phase)
            instrumental_spec = pred_instrumental_mag * torch.exp(1j * chunk_phase)
            
            # Reconstruct audio
            vocal_chunk = torch.zeros_like(chunk)
            inst_chunk = torch.zeros_like(chunk)
            for channel in range(2):
                vocal_chunk[channel] = torch.istft(
                    vocal_spec[channel].unsqueeze(0),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    length=chunk_length,
                    return_complex=False
                ).squeeze(0)
                
                inst_chunk[channel] = torch.istft(
                    instrumental_spec[channel].unsqueeze(0),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    length=chunk_length,
                    return_complex=False
                ).squeeze(0)
            
            # Handle overlap-add
            if i == 0:
                # First chunk - just copy
                copy_length = min(chunk_length, total_length)
                vocals[:, :copy_length] = vocal_chunk[:, :copy_length]
                instrumentals[:, :copy_length] = inst_chunk[:, :copy_length]
            else:
                # Cross-fade with previous chunk
                fade_in = torch.linspace(0, 1, cross_fade_length).to(device)
                fade_out = torch.linspace(1, 0, cross_fade_length).to(device)
                
                # Determine actual overlap region
                overlap_start = i
                overlap_end = min(i + cross_fade_length, total_length)
                actual_overlap = overlap_end - overlap_start
                
                if actual_overlap > 0:
                    # Vocals cross-fade
                    vocal_chunk[:, :actual_overlap] *= fade_in[:actual_overlap]
                    vocals[:, overlap_start:overlap_end] *= fade_out[:actual_overlap]
                    vocals[:, overlap_start:overlap_end] += vocal_chunk[:, :actual_overlap]
                    
                    # Instrumentals cross-fade
                    inst_chunk[:, :actual_overlap] *= fade_in[:actual_overlap]
                    instrumentals[:, overlap_start:overlap_end] *= fade_out[:actual_overlap]
                    instrumentals[:, overlap_start:overlap_end] += inst_chunk[:, :actual_overlap]
                
                # Copy remaining samples
                remaining_start = min(i + cross_fade_length, total_length)
                remaining_end = min(i + chunk_length, total_length)
                if remaining_start < remaining_end:
                    vocals[:, remaining_start:remaining_end] = vocal_chunk[:, remaining_start-i:remaining_end-i]
                    instrumentals[:, remaining_start:remaining_end] = inst_chunk[:, remaining_start-i:remaining_end-i]
            
            pbar.update(1)
    
    # Trim to original length and clamp
    vocals = vocals[:, :total_length].clamp(-1.0, 1.0)
    instrumentals = instrumentals[:, :total_length].clamp(-1.0, 1.0)
    
    torchaudio.save(output_vocal_path, vocals.cpu(), sr)
    torchaudio.save(output_instrumental_path, instrumentals.cpu(), sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for instrumental separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='train', help='Path to training dataset')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint_steps', type=int, default=2000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--segment_length', type=int, default=88200, help='Segment length for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for the optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel()
    optimizer = torch.optim.Adam(model.parameters())

    if args.train:
        train_dataset = MUSDBDataset(root_dir=args.data_dir,
                                     segment_length=args.segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=16, pin_memory=False, persistent_workers=True)
        total_steps = args.epochs * len(train_dataloader)
        train(model, train_dataloader, optimizer, loss_fn, device, args.epochs, args.checkpoint_steps, args, checkpoint_path=args.checkpoint_path, window=window)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
