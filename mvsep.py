import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from neuralop.models import FNO
import auraloss
import numpy as np
import math
import glob
from torch.utils.checkpoint import checkpoint

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=40, n_modes=(32, 32)):
        super(NeuralModel, self).__init__()
        
        # Projection with harmonic awareness
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            HarmonicAwareBlock(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.proj_skip = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # Harmonic suppression branch
        self.harmonic_suppressor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            HarmonicSuppression(hidden_channels),
            nn.GELU()
        )
        
        # Multi-scale processing with frequency discrimination
        self.low_band = nn.Sequential(
            nn.AvgPool2d((8, 1)),
            *[ResBlock(hidden_channels) for _ in range(2)],
            SpectralDiscriminationBlock(hidden_channels),
            nn.Upsample(scale_factor=(8, 1), mode='bilinear', align_corners=False)
        )
        
        self.mid_band = nn.Sequential(
            *[ResBlock(hidden_channels) for _ in range(3)],
            VocalCharacteristicAttention(hidden_channels)  # Added attention
        )
        
        self.high_band = nn.Sequential(
            nn.MaxPool2d((1, 4)),
            *[ResBlock(hidden_channels) for _ in range(2)],
            SpectralDiscriminationBlock(hidden_channels),
            nn.Upsample(scale_factor=(1, 4), mode='bilinear', align_corners=False)
        )
        
        # Sub-band processing with harmonic suppression
        self.sub_bands = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels//4, 3, padding=1),
                HarmonicSuppression(hidden_channels//4),
                nn.GELU()
            ) for _ in range(4)
        ])
        
        # Frequency attention
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(hidden_channels, hidden_channels//4, 1),
            nn.GELU(),
            VocalCharacteristicAttention(hidden_channels//4),
            nn.Conv2d(hidden_channels//4, hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Time-scale processing with vocal characteristics
        self.slow_path = nn.Sequential(
            nn.AvgPool2d((1, 8)),
            *[ResBlock(hidden_channels) for _ in range(2)],
            VocalCharacteristicAttention(hidden_channels),
            nn.Upsample(scale_factor=(1, 8), mode='bilinear', align_corners=False)
        )
        
        # Phase-aware processing
        self.phase_aware = nn.Sequential(
            nn.Conv2d(1, hidden_channels//4, 3, padding=1),
            HarmonicAwareBlock(hidden_channels//4),
            nn.GELU(),
            nn.Conv2d(hidden_channels//4, hidden_channels, 3, padding=1)
        )
        
        # Feature combiner with dynamic weighting
        self.combiner = nn.Sequential(
            nn.Conv2d(hidden_channels*6, hidden_channels*2, 1),
            nn.GELU(),
            VocalCharacteristicEnhancement(hidden_channels*2),
            nn.Conv2d(hidden_channels*2, hidden_channels, 1)
        )
        self.branch_weights = nn.Parameter(torch.ones(6))
        
        # FNO with residual connection
        self.operator = FNO(n_modes=n_modes, 
                          hidden_channels=hidden_channels,
                          in_channels=hidden_channels, 
                          out_channels=hidden_channels)
        self.operator_residual = nn.Conv2d(hidden_channels, hidden_channels, 1)
        
        # Enhanced mask predictor
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels*2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels*2),
            nn.GELU(),
            VocalCharacteristicAttention(hidden_channels*2),
            nn.Conv2d(hidden_channels*2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels//2, 3, padding=1),
            SpectralDiscriminationBlock(hidden_channels//2),
            nn.GELU(),
            nn.Conv2d(hidden_channels//2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Projection with skip and harmonic suppression
        x_proj = self.projection(x) + self.proj_skip(x)
        x_proj = self.harmonic_suppressor(x_proj)
        
        # Get target size from mid band (reference size)
        x_mid = self.mid_band(x_proj)
        target_size = x_mid.shape[2:]
        
        # Multi-scale processing with size adjustment
        x_low = self.low_band(x_proj)
        x_low = F.interpolate(x_low, size=target_size, mode='bilinear', align_corners=False)
        
        x_high = self.high_band(x_proj)
        x_high = F.interpolate(x_high, size=target_size, mode='bilinear', align_corners=False)
        
        # Sub-band processing with size adjustment
        sub_outs = []
        chunk_size = x_proj.shape[2] // 4
        for i, sub_conv in enumerate(self.sub_bands):
            start = i * chunk_size
            end = (i+1) * chunk_size if i < 3 else x_proj.shape[2]
            sub = x_proj[:, :, start:end, :]
            sub_out = sub_conv(sub)
            sub_out = F.interpolate(sub_out, size=target_size, mode='bilinear', align_corners=False)
            sub_outs.append(sub_out)
        x_sub = torch.cat(sub_outs, dim=1)
        x_sub = F.interpolate(x_sub, size=target_size, mode='bilinear', align_corners=False)
        
        # Time-scale processing with size adjustment
        x_slow = self.slow_path(x_proj)
        x_slow = F.interpolate(x_slow, size=target_size, mode='bilinear', align_corners=False)
        
        # Phase-aware processing
        if x.dim() == 4 and x.shape[1] == 2:  # If we have phase info
            phase = torch.angle(torch.view_as_complex(x.permute(0,2,3,1).contiguous()))
            phase_feat = self.phase_aware(phase.unsqueeze(1))
            phase_feat = F.interpolate(phase_feat, size=target_size, mode='bilinear', align_corners=False)
        else:
            phase_feat = 0
        
        # Frequency attention with size adjustment
        x_attn = self.freq_attention(x_proj)
        x_attn = F.interpolate(x_attn, size=target_size, mode='bilinear', align_corners=False)
        
        # Combine all branches with learned weights
        branches = [x_low, x_mid, x_high, x_sub, x_slow, x_attn]
        
        # Ensure all branches have exactly the same size
        for i in range(len(branches)):
            if branches[i].shape[2:] != target_size:
                branches[i] = F.interpolate(branches[i], size=target_size, 
                                          mode='bilinear', align_corners=False)
        
        weights = F.softmax(self.branch_weights, dim=0)
        x_combined = sum(w*b for w,b in zip(weights, branches)) + phase_feat
        
        # Final combination
        x = self.combiner(torch.cat(branches, dim=1))
        
        # FNO processing with residual
        x_operator = checkpoint(self.operator, x, use_reentrant=False)
        x = x + self.operator_residual(x_operator)
        
        # Final mask prediction
        vocal_mask = self.mask_predictor(x + phase_feat)
        return vocal_mask.expand(-1, 2, -1, -1)

class HarmonicAwareBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        # Dynamically pick the largest possible num_groups ≤ 8 that divides channels
        possible_groups = [g for g in range(8, 0, -1) if channels % g == 0]
        num_groups = possible_groups[0] if possible_groups else 1  # fallback to 1
        self.norm = nn.GroupNorm(num_groups, channels)
    def forward(self, x):
        x = F.gelu(self.norm(self.conv1(x)))
        x = F.gelu(self.norm(self.conv2(x)))
        return x

class HarmonicSuppression(nn.Module):
    """Helps suppress harmonic instruments while preserving vocals"""
    def __init__(self, channels):
        super().__init__()
        self.freq_conv = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, channels//4, 1),
            nn.GELU(),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.freq_conv(x)
        gate = self.gate(x)
        return x * (1 - gate) + h * gate  # Adaptive suppression

class VocalCharacteristicAttention(nn.Module):
    """Focuses on vocal characteristics like vibrato and formants"""
    def __init__(self, channels):
        super().__init__()
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(channels, channels//4, (1, 5), padding=(0, 2)),
            nn.GELU(),
            nn.Conv2d(channels//4, channels, (1, 5), padding=(0, 2)),
            nn.Sigmoid()
        )
        self.freq_attn = nn.Sequential(
            nn.Conv2d(channels, channels//4, (5, 1), padding=(2, 0)),
            nn.GELU(),
            nn.Conv2d(channels//4, channels, (5, 1), padding=(2, 0)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        t_attn = self.temporal_attn(x)  # Capture vibrato/amplitude modulation
        f_attn = self.freq_attn(x)      # Capture formant structure
        return x * t_attn * f_attn

class SpectralDiscriminationBlock(nn.Module):
    """Helps discriminate between vocal and instrumental spectra"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, channels//4, 1),
            nn.GELU(),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = F.gelu(self.conv(x))
        gate = self.spectral_gate(x)
        return x * gate

class VocalCharacteristicEnhancement(nn.Module):
    """Enhances vocal characteristics in combined features"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.attn = VocalCharacteristicAttention(channels)
        
    def forward(self, x):
        x = F.gelu(self.conv(x))
        return self.attn(x)

class ResBlock(nn.Module):
    """Helper residual block for better gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual

def loss_fn(pred_vocal_mask,
            target_vocal_mag,
            mixture_mag, mixture_phase,
            window, n_fft, hop_length):

    auraloss1 = auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        perceptual_weighting=True,
        sample_rate=44100,
        scale="mel",
        n_bins=128,
        device="cuda"
    )
    
    # Apply the predicted mask to the mixture magnitude.
    pred_vocal_mag = mixture_mag * pred_vocal_mask  # [B, 2, F, T]

    # Helper function to create a complex spectrogram.
    def make_complex(mag):
        return mag * torch.exp(1j * mixture_phase)

    # Convert predicted and target magnitudes to complex spectrograms.
    pred_vocal_spec = make_complex(pred_vocal_mag)
    target_vocal_spec = make_complex(target_vocal_mag)

    # ISTFT helper for batched channels.
    def istft_channels(spec):
        batch_size, channels, F, T = spec.shape
        spec_combined = spec.reshape(batch_size * channels, F, T)
        audio = torch.istft(spec_combined, n_fft=n_fft, hop_length=hop_length, window=window)
        audio = audio.reshape(batch_size, channels, -1)
        return audio

    pred_vocal_audio = istft_channels(pred_vocal_spec)  # [B, 2, L]
    target_vocal_audio = istft_channels(target_vocal_spec)
    
    total_loss = auraloss1(pred_vocal_audio, target_vocal_audio)
    return total_loss

class MUSDBDataset(Dataset):
    def __init__(self, root_dir, preprocess_dir=None, sample_rate=44100, segment_length=485100, segment=True):
        self.root_dir = root_dir
        self.preprocess_dir = preprocess_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = 4096
        self.hop_length = 1024
        self.segment = segment
        self.tracks = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]
        self.window = torch.hann_window(self.n_fft)

        if self.preprocess_dir:
            self.preprocess_data()

    def preprocess_data(self):
        os.makedirs(self.preprocess_dir, exist_ok=True)
        for idx, track_path in enumerate(tqdm(self.tracks, desc="Preprocessing data")):
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            if not os.path.exists(preprocess_path):
                mixture_mag, mixture_phase, vocal_mag, vocal_phase = self._process_track(track_path)
                np.savez(preprocess_path, mixture_mag=mixture_mag, mixture_phase=mixture_phase,
                         vocal_mag=vocal_mag, vocal_phase=vocal_phase)

    def _process_track(self, track_path):
        vocal, _ = torchaudio.load(os.path.join(track_path, 'vocals.wav'))

        if vocal.shape[0] != 2:
            raise ValueError("Audio file must have 2 channels.")
        
        mixture = vocal.clone()
        
        mixture_spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        vocal_spec = torch.stft(vocal, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)

        mixture_mag = torch.abs(mixture_spec)
        mixture_phase = torch.angle(mixture_spec)
        vocal_mag = torch.abs(vocal_spec)
        vocal_phase = torch.angle(vocal_spec)

        if self.segment and self.segment_length:
            # Calculate number of time frames corresponding to the segment length.
            num_frames = self.segment_length // self.hop_length
            if mixture_mag.shape[2] >= num_frames:
                start = torch.randint(0, mixture_mag.shape[2] - num_frames, (1,))
                mixture_mag = mixture_mag[:, :, start:start + num_frames]
                mixture_phase = mixture_phase[:, :, start:start + num_frames]
                vocal_mag = vocal_mag[:, :, start:start + num_frames]
                vocal_phase = vocal_phase[:, :, start:start + num_frames]
            else:
                pad_amount = num_frames - mixture_mag.shape[2]
                mixture_mag = F.pad(mixture_mag, (0, pad_amount))
                mixture_phase = F.pad(mixture_phase, (0, pad_amount))
                vocal_mag = F.pad(vocal_mag, (0, pad_amount))
                vocal_phase = F.pad(vocal_phase, (0, pad_amount))

        return (mixture_mag.numpy(), mixture_phase.numpy(),
                vocal_mag.numpy(), vocal_phase.numpy())

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if self.preprocess_dir:
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            data = np.load(preprocess_path)
            mixture_mag = torch.from_numpy(data['mixture_mag'])
            mixture_phase = torch.from_numpy(data['mixture_phase'])
            vocal_mag = torch.from_numpy(data['vocal_mag'])
            vocal_phase = torch.from_numpy(data['vocal_phase'])
            return mixture_mag, mixture_phase, vocal_mag, vocal_phase
        else:
            return self._process_track(self.tracks[idx])

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
            mixture_mag, mixture_phase, vocal_mag, vocal_phase = batch
            mixture_mag = mixture_mag.to(device)
            mixture_phase = mixture_phase.to(device)
            vocal_mag = vocal_mag.to(device)

            optimizer.zero_grad()
            # Predict the vocal mask from the mixture magnitude.
            pred_vocal_mask = model(mixture_mag)
            loss = loss_fn(pred_vocal_mask, vocal_mag, mixture_mag, mixture_phase, window, 4096, 1024)
            
            # Replace NaN loss with zero.
            if torch.isnan(loss).any():
                print("NaN loss detected, replacing with 0.")
                loss = torch.zeros_like(loss)

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
    
    # Load audio and check sample rate
    input_audio, sr = torchaudio.load(input_wav_path)
    if sr != 44100:
        raise ValueError(f"Input audio must be 44100Hz, but got {sr}Hz. Please resample the audio first.")
    if input_audio.shape[0] != 2:
        raise ValueError("Input audio must have 2 channels.")
        
    input_audio = input_audio.to(device)
    total_length = input_audio.shape[1]
    vocals = torch.zeros_like(input_audio)
    instrumentals = torch.zeros_like(input_audio)
    cross_fade_length = overlap // 2
    window = torch.hann_window(4096).to(device)
    num_chunks = (total_length - overlap) // (chunk_size - overlap)
    
    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length - chunk_size + 1, chunk_size - overlap):
            chunk = input_audio[:, i:i + chunk_size]
            chunk_spec = torch.stft(chunk, n_fft=4096, hop_length=1024, window=window, return_complex=True)
            chunk_mag = torch.abs(chunk_spec)
            chunk_phase = torch.angle(chunk_spec)
            # Add batch dimension and channel dimension if necessary.
            chunk_mag = chunk_mag.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_vocal_mask = model(chunk_mag)  # [1, 2, F, T]
            # Squeeze batch dimension.
            pred_vocal_mask = pred_vocal_mask.squeeze(0)
            # Multiply with chunk magnitude.
            pred_vocal_mag = chunk_mag.squeeze(0) * pred_vocal_mask
            # Reconstruct vocal spectrogram.
            pred_vocal_spec = pred_vocal_mag * torch.exp(1j * chunk_phase)
            
            # Reconstruct vocal and then compute instrumental = mixture - vocals.
            vocal_chunk = torch.zeros_like(chunk)
            for channel in range(2):
                vocal_chunk[channel] = torch.istft(
                    pred_vocal_spec[channel].unsqueeze(0),
                    n_fft=4096,
                    hop_length=1024,
                    window=window,
                    length=chunk_size,
                    return_complex=False
                ).squeeze(0)
            inst_chunk = chunk - vocal_chunk

            # Apply cross-fading for smooth overlap-add.
            if i == 0:
                vocals[:, i:i + chunk_size] = vocal_chunk
                instrumentals[:, i:i + chunk_size] = inst_chunk
            else:
                fade_in = torch.linspace(0, 1, cross_fade_length).to(device)
                fade_out = torch.linspace(1, 0, cross_fade_length).to(device)
                vocal_chunk[:, :cross_fade_length] *= fade_in
                vocals[:, i:i + cross_fade_length] *= fade_out
                vocals[:, i:i + cross_fade_length] += vocal_chunk[:, :cross_fade_length]
                
                inst_chunk[:, :cross_fade_length] *= fade_in
                instrumentals[:, i:i + cross_fade_length] *= fade_out
                instrumentals[:, i:i + cross_fade_length] += inst_chunk[:, :cross_fade_length]
                
                vocals[:, i + cross_fade_length:i + chunk_size] = vocal_chunk[:, cross_fade_length:]
                instrumentals[:, i + cross_fade_length:i + chunk_size] = inst_chunk[:, cross_fade_length:]
            pbar.update(1)
    vocals = torch.clamp(vocals, -1.0, 1.0)
    instrumentals = torch.clamp(instrumentals, -1.0, 1.0)
    torchaudio.save(output_vocal_path, vocals.cpu(), sr)
    torchaudio.save(output_instrumental_path, instrumentals.cpu(), sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for instrumental separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='train', help='Path to training dataset')
    parser.add_argument('--preprocess_dir', type=str, default='prep', help='Path to save/load preprocessed data')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint_steps', type=int, default=2000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--segment_length', type=int, default=485100, help='Segment length for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for the optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel()
    optimizer = torch.optim.Adam(model.parameters())

    if args.train:
        train_dataset = MUSDBDataset(root_dir=args.data_dir, preprocess_dir=args.preprocess_dir,
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
