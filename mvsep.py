import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prodigyopt import Prodigy
import numpy as np
import random
import math
import glob
from torch.utils.checkpoint import checkpoint
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

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
    def __init__(self, dim, in_channels=None, max_freq=10000, init_scale=1.0, mlp_hidden=128):
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

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = UNetConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels includes channels from skip connection + upsampled features
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = UNetConvBlock(in_channels, out_channels, in_channels // 2)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch due to pooling/convolutions
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=128, out_channels=2):
        super(NeuralModel, self).__init__()
        
        # U1 Encoder Path
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        )
        
        # Transformer configurations
        encoder_layer = lambda d: TransformerEncoderLayer(d_model=d, nhead=8, batch_first=True)
        decoder_layer = lambda d: TransformerDecoderLayer(d_model=d, nhead=8, batch_first=True)
        
        # U1 Path Transformers + DARPE
        self.darpe_proj = DARPE(dim=hidden_channels)
        self.gru_proj = TransformerEncoder(encoder_layer(hidden_channels), num_layers=1)
        
        self.down1 = DownBlock(hidden_channels, hidden_channels*2)
        self.darpe_down1 = DARPE(dim=hidden_channels*2)
        self.gru_down1 = TransformerEncoder(encoder_layer(hidden_channels*2), num_layers=1)
        
        self.down2 = DownBlock(hidden_channels*2, hidden_channels*4)
        self.darpe_down2 = DARPE(dim=hidden_channels*4)
        self.gru_down2 = TransformerEncoder(encoder_layer(hidden_channels*4), num_layers=1)
        
        # U1 Center
        self.center_conv = UNetConvBlock(hidden_channels*4, hidden_channels*4)
        self.darpe_center = DARPE(dim=hidden_channels*4)
        self.gru_center = TransformerEncoder(encoder_layer(hidden_channels*4), num_layers=1)
        
        # U1 Decoder Path
        self.up1 = UpBlock(hidden_channels*4 + hidden_channels*4, hidden_channels*2)
        self.darpe_up1 = DARPE(dim=hidden_channels*2)
        self.gru_up1 = TransformerDecoder(decoder_layer(hidden_channels*2), num_layers=1)
        
        self.up2 = UpBlock(hidden_channels*2 + hidden_channels*2, hidden_channels)
        self.darpe_up2 = DARPE(dim=hidden_channels)
        self.gru_up2 = TransformerDecoder(decoder_layer(hidden_channels), num_layers=1)
        
        # U2 Path
        self.down3 = DownBlock(hidden_channels, hidden_channels*2)
        self.darpe_down3 = DARPE(dim=hidden_channels*2)
        self.gru_down3 = TransformerEncoder(encoder_layer(hidden_channels*2), num_layers=1)
        
        self.down4 = DownBlock(hidden_channels*2, hidden_channels*4)
        self.darpe_down4 = DARPE(dim=hidden_channels*4)
        self.gru_down4 = TransformerEncoder(encoder_layer(hidden_channels*4), num_layers=1)
        
        # U2 Center
        self.center2_conv = UNetConvBlock(hidden_channels*4, hidden_channels*4)
        self.darpe_center2 = DARPE(dim=hidden_channels*4)
        self.gru_center2 = TransformerEncoder(encoder_layer(hidden_channels*4), num_layers=1)
        
        # U2 Decoder Path
        self.up3 = UpBlock(hidden_channels*4 + hidden_channels*4, hidden_channels*2)
        self.darpe_up3 = DARPE(dim=hidden_channels*2)
        self.gru_up3 = TransformerDecoder(decoder_layer(hidden_channels*2), num_layers=1)
        
        self.up4 = UpBlock(hidden_channels*2 + hidden_channels*2, hidden_channels)
        self.darpe_up4 = DARPE(dim=hidden_channels)
        self.gru_up4 = TransformerDecoder(decoder_layer(hidden_channels), num_layers=1)
        
        # Final Output
        self.final_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # U1 Processing
        x = self.projection(x)
        B, C, H, W = x.shape
        
        # Encoder path
        x = self.darpe_proj(x)  # DARPE before Transformer
        x = self.apply_transformer(self.gru_proj, x)
        
        skip1, down1 = self.down1(x)
        down1 = self.darpe_down1(down1)
        down1 = self.apply_transformer(self.gru_down1, down1)
        
        skip2, down2 = self.down2(down1)
        down2 = self.darpe_down2(down2)
        down2 = self.apply_transformer(self.gru_down2, down2)
        
        # Center
        center = self.center_conv(down2)
        center = self.darpe_center(center)
        center = self.apply_transformer(self.gru_center, center)
        
        # Decoder path with cross-attention
        up1 = self.up1(center, skip2)
        up1 = self.darpe_up1(up1)
        up1 = self.apply_decoder(self.gru_up1, up1, skip2)
        
        up2 = self.up2(up1, skip1)
        up2 = self.darpe_up2(up2)
        up2 = self.apply_decoder(self.gru_up2, up2, skip1)
        
        # U2 Processing
        skip3, down3 = self.down3(up2)
        down3 = self.darpe_down3(down3)
        down3 = self.apply_transformer(self.gru_down3, down3)
        
        skip4, down4 = self.down4(down3)
        down4 = self.darpe_down4(down4)
        down4 = self.apply_transformer(self.gru_down4, down4)
        
        # Center
        center2 = self.center2_conv(down4)
        center2 = self.darpe_center2(center2)
        center2 = self.apply_transformer(self.gru_center2, center2)
        
        # Decoder path
        up3 = self.up3(center2, skip4)
        up3 = self.darpe_up3(up3)
        up3 = self.apply_decoder(self.gru_up3, up3, skip4)
        
        up4 = self.up4(up3, skip3)
        up4 = self.darpe_up4(up4)
        up4 = self.apply_decoder(self.gru_up4, up4, skip3)
        
        # Final Output
        pred_vocal_mask = self.final_proj(up4)
        final_vocal_mask = self.final_activation(pred_vocal_mask)
        return final_vocal_mask

    def apply_transformer(self, transformer_layer, x):
        """For encoder layers"""
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        x_transformed = transformer_layer(x_flat)
        x_out = x_transformed.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x_out

    def apply_decoder(self, decoder_layer, x, memory):
        """For decoder layers with cross-attention"""
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        memory_flat = memory.permute(0, 2, 3, 1).reshape(B * H, -1, C)
        
        # Process through decoder
        x_decoded = decoder_layer(x_flat, memory_flat)
        
        x_out = x_decoded.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x_out

def aweight_coefficients(freqs):
    freqs = np.array(freqs)
    # avoid zero-frequency singularity
    freqs = np.maximum(freqs, 1e-6)
    numer = (12194**2) * (freqs**4)
    denom = (freqs**2 + 20.6**2) * np.sqrt(
        (freqs**2 + 107.7**2) * (freqs**2 + 737.9**2)
    ) * (freqs**2 + 12194**2)
    return 20 * np.log10(numer / (denom + 1e-10)) + 2.0

def loss_fn(pred_vocal_mask, target_vocal_mag, target_instrumental_mag, mixture_mag,
            scales=None, weights=None):
    device = mixture_mag.device
    dtype = mixture_mag.dtype

    # STFT params
    n_fft = 4096
    sample_rate = 44100

    # frequency bins
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    a_weights = aweight_coefficients(freqs)           # [2049]
    a_weights = torch.tensor(a_weights, device=device, dtype=dtype)
    a_weights = a_weights.view(1, 1, -1, 1)            # [1,1,2049,1]
    a_weights_linear = torch.pow(10.0, a_weights / 20.0)

    mse = torch.nn.MSELoss()
    total_loss = 0.0

    if scales  is None: scales  = [1, 2, 4, 8]
    if weights is None: weights = [1.0] * len(scales)

    for scale, w in zip(scales, weights):
        # downsample everything (including masks) except at scale=1
        if scale == 1:
            pm, mix, tv, ti = (
                pred_vocal_mask,
                mixture_mag,
                target_vocal_mag,
                target_instrumental_mag
            )
            aw = a_weights_linear
        else:
            size = (
                pred_vocal_mask.shape[2] // scale,
                pred_vocal_mask.shape[3] // scale
            )
            pm  = F.interpolate(pred_vocal_mask,        size=size, mode='area')
            mix = F.interpolate(mixture_mag,             size=size, mode='area')
            tv  = F.interpolate(target_vocal_mag,        size=size, mode='area')
            ti  = F.interpolate(target_instrumental_mag, size=size, mode='area')

            # downsample A-weights to [1,1,freq_bins,1]
            aw = F.interpolate(
                a_weights_linear,
                size=(pm.shape[2], 1),
                mode='area'
            )

        # apply perceptual weighting
        pv_weighted = (mix * pm)         * aw
        pi_weighted = (mix * (1.0 - pm)) * aw
        tv_weighted = tv                * aw
        ti_weighted = ti                * aw

        loss_v = mse(pv_weighted, tv_weighted)
        loss_i = mse(pi_weighted, ti_weighted)
        total_loss += w * (loss_v + loss_i)

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

    total_steps = epochs * len(dataloader)
    progress_bar = tqdm(total=total_steps, initial=step)
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
            loss = loss_fn(pred_vocal_mask, vocal_mag, instrumental_mag, mixture_mag)
            
            if torch.isnan(loss).any():
                print("NaN loss detected, skipping batch")
                continue

            loss.backward()
            optimizer.step()

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            step += 1
            progress_bar.update(1)
            current_lr = optimizer.param_groups[0]['lr']
            desc = f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f}"
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
              chunk_size=32000, overlap=16000, device='cpu'):
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
    parser.add_argument('--segment_length', type=int, default=32000, help='Segment length for training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel()
    optimizer = Prodigy(model.parameters(), lr=1.0)

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
