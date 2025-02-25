import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch_log_wmse import LogWMSE
import math
import glob
from torch.utils.checkpoint import checkpoint

class RotaryEmbedding3D(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even for rotary embedding")
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, pos):
        # Compute the rotary angles: (N, dim//2)
        theta = pos.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        sin, cos = theta.sin(), theta.cos()
        # Reshape x into pairs: (N, dim//2, 2)
        x = x.view(-1, self.dim // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        # Apply the rotation for each pair
        x_rotated_0 = x1 * cos - x2 * sin
        x_rotated_1 = x1 * sin + x2 * cos
        x_rot = torch.stack([x_rotated_0, x_rotated_1], dim=-1)
        return x_rot.view(-1, self.dim)

class RotaryPositionalEmbedding3D(nn.Module):
    def __init__(self, hidden_channels, base=10000):
        super().__init__()
        if hidden_channels % 3 != 0:
            raise ValueError("hidden_channels must be divisible by 3 for 3D rotary embedding")
        self.hidden_channels = hidden_channels
        self.dim_each = hidden_channels // 3
        if self.dim_each % 2 != 0:
            raise ValueError("hidden_channels/3 must be even for rotary embedding")
        self.rotary_time = RotaryEmbedding3D(self.dim_each, base)   # Time dimension
        self.rotary_mag = RotaryEmbedding3D(self.dim_each, base)    # Magnitude dimension
        self.rotary_phase = RotaryEmbedding3D(self.dim_each, base)  # Phase dimension

    def forward(self, x):
        B, C, H, W = x.shape  # Assume: H = number of frequency bins, W = number of time steps
        # Split channels into three parts: time, magnitude, phase
        x_time, x_mag, x_phase = torch.chunk(x, 3, dim=1)  # each is (B, C/3, H, W)

        # Create coordinate grids for each dimension:
        # For time (W axis)
        pos_time = torch.arange(W, device=x.device).float().view(1, 1, 1, W)  # (1,1,1,W)
        # For magnitude (H axis)
        pos_mag = torch.arange(H, device=x.device).float().view(1, 1, H, 1)   # (1,1,H,1)
        # For phase, we generate a distinct coordinate grid.
        # For example, let phase vary from -pi to pi over H bins.
        pos_phase = torch.linspace(-math.pi, math.pi, H, device=x.device).float().view(1, 1, H, 1)

        # Bring the channel dimension to the end:
        # Now each x_* is (B, H, W, dim_each)
        x_time = x_time.permute(0, 2, 3, 1)
        x_mag = x_mag.permute(0, 2, 3, 1)
        x_phase = x_phase.permute(0, 2, 3, 1)

        # Flatten the (B, H, W) dimensions:
        x_time_flat = x_time.reshape(-1, self.dim_each)
        x_mag_flat = x_mag.reshape(-1, self.dim_each)
        x_phase_flat = x_phase.reshape(-1, self.dim_each)

        # Expand and flatten the position tensors to match (B*H*W,)
        pos_time_flat = pos_time.expand(B, 1, H, W).reshape(-1)
        pos_mag_flat = pos_mag.expand(B, 1, H, W).reshape(-1)
        pos_phase_flat = pos_phase.expand(B, 1, H, W).reshape(-1)

        # Apply rotary embedding on each part
        x_time_rot = self.rotary_time(x_time_flat, pos_time_flat)
        x_mag_rot = self.rotary_mag(x_mag_flat, pos_mag_flat)
        x_phase_rot = self.rotary_phase(x_phase_flat, pos_phase_flat)

        # Reshape back to (B, H, W, dim_each)
        x_time_rot = x_time_rot.view(B, H, W, self.dim_each)
        x_mag_rot = x_mag_rot.view(B, H, W, self.dim_each)
        x_phase_rot = x_phase_rot.view(B, H, W, self.dim_each)

        # Concatenate along the channel dimension and permute back to (B, hidden_channels, H, W)
        x_out = torch.cat([x_time_rot, x_mag_rot, x_phase_rot], dim=-1)  # (B, H, W, hidden_channels)
        x_out = x_out.permute(0, 3, 1, 2)
        return x_out

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=128, num_layers=1):
        super(NeuralModel, self).__init__()
        self.projection = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        self.rotary_pos_emb = RotaryPositionalEmbedding3D(hidden_channels)
        
        self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels,
                            num_layers=num_layers, batch_first=True)
        
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: shape (B, in_channels, H, W)
        x = self.projection(x)
        
        x = self.rotary_pos_emb(x)
        
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1)  # now (B, F, T, C)
        x = x.reshape(B * F, T, C)  # merge batch and frequency dims

        # Wrap the LSTM forward pass using a lambda that returns only the output.
        x = checkpoint(lambda inp: self.lstm(inp)[0], x, use_reentrant=False)
        
        # Reshape back to (B, F, T, C) then permute to (B, C, F, T)
        x = x.reshape(B, F, T, C)
        x = x.permute(0, 3, 1, 2)
        
        mask = self.mask_predictor(x)
        vocal_mask, inst_mask = torch.split(mask, 1, dim=1)
        return inst_mask, vocal_mask

def loss_fn(pred_inst_mask, pred_vocal_mask,
            target_inst_mag, target_vocal_mag,
            mixture_mag, mixture_phase,
            window, n_fft, hop_length):
    # Remove batch dimension (assuming batch_size=1)
    pred_inst_mask = pred_inst_mask.squeeze(0)  # [channels, F, T]
    pred_vocal_mask = pred_vocal_mask.squeeze(0)
    target_inst_mag = target_inst_mag.squeeze(0)
    target_vocal_mag = target_vocal_mag.squeeze(0)
    mixture_mag = mixture_mag.squeeze(0)
    mixture_phase = mixture_phase.squeeze(0)

    # Apply masks to mixture magnitude
    pred_inst_mag = mixture_mag * pred_inst_mask
    pred_vocal_mag = mixture_mag * pred_vocal_mask

    # Reconstruct complex spectrograms
    def make_complex(mag):
        return mag * torch.exp(1j * mixture_phase)
    
    pred_inst_spec = make_complex(pred_inst_mag)
    pred_vocal_spec = make_complex(pred_vocal_mag)
    target_inst_spec = make_complex(target_inst_mag)
    target_vocal_spec = make_complex(target_vocal_mag)
    mixture_spec = make_complex(mixture_mag)

    # ISTFT function for each channel
    def istft_channels(spec):
        return torch.stack([
            torch.istft(spec[ch], n_fft=n_fft, hop_length=hop_length, window=window)
            for ch in range(spec.shape[0])
        ], dim=0)

    # Convert to time-domain audio
    pred_inst_audio = istft_channels(pred_inst_spec)  # [channels, time]
    pred_vocal_audio = istft_channels(pred_vocal_spec)
    target_inst_audio = istft_channels(target_inst_spec)
    target_vocal_audio = istft_channels(target_vocal_spec)
    mixture_audio = istft_channels(mixture_spec)  # [channels, time]

    # Add batch dimension
    pred_inst_audio = pred_inst_audio.unsqueeze(0)  # [1, channels, time]
    pred_vocal_audio = pred_vocal_audio.unsqueeze(0)
    target_inst_audio = target_inst_audio.unsqueeze(0)
    target_vocal_audio = target_vocal_audio.unsqueeze(0)
    mixture_audio = mixture_audio.unsqueeze(0)

    # Format for LogWMSE: [batch, stems, channels, time]
    processed_audio = torch.stack([pred_inst_audio, pred_vocal_audio], dim=1)  # [1, 2, 2, time]
    target_audio = torch.stack([target_inst_audio, target_vocal_audio], dim=1)

    # Initialize losses
    log_wmse = LogWMSE(
        audio_length=pred_inst_audio.shape[-1]/44100,
        sample_rate=44100,
        return_as_loss=True,
        bypass_filter=False
    )

    # Compute LogWMSE loss
    logwmse_loss = log_wmse(mixture_audio, processed_audio, target_audio)
    total_loss = logwmse_loss

    return total_loss

# Custom Dataset class
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
                (mixture_mag, mixture_phase, instrumental_mag, instrumental_phase,
                 vocal_mag, vocal_phase, _, _, _) = self._process_track(track_path)
                np.savez(preprocess_path, mixture_mag=mixture_mag, mixture_phase=mixture_phase,
                         instrumental_mag=instrumental_mag, instrumental_phase=instrumental_phase,
                         vocal_mag=vocal_mag, vocal_phase=vocal_phase)

    def _process_track(self, track_path):
        instrumental, _ = torchaudio.load(os.path.join(track_path, 'other.wav'))
        vocal, _ = torchaudio.load(os.path.join(track_path, 'vocals.wav'))

        if instrumental.shape[0] != 2 or vocal.shape[0] != 2:
            raise ValueError("Audio files must have 2 channels.")

        min_length = min(instrumental.shape[1], vocal.shape[1])
        instrumental = instrumental[:, :min_length]
        vocal = vocal[:, :min_length]
        mixture = instrumental + vocal

        mixture_spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        instrumental_spec = torch.stft(instrumental, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        vocal_spec = torch.stft(vocal, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)

        mixture_mag = torch.abs(mixture_spec)
        mixture_phase = torch.angle(mixture_spec)
        instrumental_mag = torch.abs(instrumental_spec)
        instrumental_phase = torch.angle(instrumental_spec)
        vocal_mag = torch.abs(vocal_spec)
        vocal_phase = torch.angle(vocal_spec)

        if self.segment and self.segment_length:
            if mixture_mag.shape[2] >= self.segment_length // self.hop_length:
                start = torch.randint(0, mixture_mag.shape[2] - self.segment_length // self.hop_length, (1,))
                mixture_mag = mixture_mag[:, :, start:start + self.segment_length // self.hop_length]
                mixture_phase = mixture_phase[:, :, start:start + self.segment_length // self.hop_length]
                instrumental_mag = instrumental_mag[:, :, start:start + self.segment_length // self.hop_length]
                instrumental_phase = instrumental_phase[:, :, start:start + self.segment_length // self.hop_length]
                vocal_mag = vocal_mag[:, :, start:start + self.segment_length // self.hop_length]
                vocal_phase = vocal_phase[:, :, start:start + self.segment_length // self.hop_length]
            else:
                pad_amount = self.segment_length // self.hop_length - mixture_mag.shape[2]
                mixture_mag = F.pad(mixture_mag, (0, pad_amount))
                mixture_phase = F.pad(mixture_phase, (0, pad_amount))
                instrumental_mag = F.pad(instrumental_mag, (0, pad_amount))
                instrumental_phase = F.pad(instrumental_phase, (0, pad_amount))
                vocal_mag = F.pad(vocal_mag, (0, pad_amount))
                vocal_phase = F.pad(vocal_phase, (0, pad_amount))

        return (mixture_mag.numpy(), mixture_phase.numpy(),
                instrumental_mag.numpy(), instrumental_phase.numpy(),
                vocal_mag.numpy(), vocal_phase.numpy(),
                mixture.numpy(), instrumental.numpy(), vocal.numpy())

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if self.preprocess_dir:
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            data = np.load(preprocess_path)
            mixture_mag = torch.from_numpy(data['mixture_mag'])
            mixture_phase = torch.from_numpy(data['mixture_phase'])
            instrumental_mag = torch.from_numpy(data['instrumental_mag'])
            instrumental_phase = torch.from_numpy(data['instrumental_phase'])
            vocal_mag = torch.from_numpy(data['vocal_mag'])
            vocal_phase = torch.from_numpy(data['vocal_phase'])
            return mixture_mag, mixture_phase, instrumental_mag, instrumental_phase, vocal_mag, vocal_phase
        else:
            track_path = self.tracks[idx]
            data = self._process_track(track_path)
            return data[:6]

def adjust_learning_rate(optimizer, grad_norm, base_lr, scale=1.0, eps=1e-8):
    grad_norm = max(grad_norm, eps)
    lr = base_lr * (1.0 / (1.0 + grad_norm / scale))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, dataloader, optimizer, scheduler, loss_fn, device, epochs, checkpoint_steps, args, checkpoint_path=None, window=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    checkpoint_files = []

    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        step = checkpoint_data['step']
        avg_loss = checkpoint_data['avg_loss']
        print(f"Resuming training from step {step} with average loss {avg_loss:.4f}")

    progress_bar = tqdm(total=epochs * len(dataloader))
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            mixture_mag, mixture_phase, instrumental_mag, _, vocal_mag, _ = batch
            mixture_mag = mixture_mag.to(device)
            mixture_phase = mixture_phase.to(device)
            instrumental_mag = instrumental_mag.to(device)
            vocal_mag = vocal_mag.to(device)

            optimizer.zero_grad()
            pred_inst_mask, pred_vocal_mask = model(mixture_mag)
            loss = loss_fn(pred_inst_mask, pred_vocal_mask, instrumental_mag, vocal_mag, mixture_mag, mixture_phase, window, 4096, 1024)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            adjust_learning_rate(optimizer, grad_norm, base_lr=args.learning_rate)
            optimizer.step()
            scheduler.step()

            if torch.isnan(loss).any():
                raise ValueError("Loss is NaN!")

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
                    'avg_loss': avg_loss
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
    input_audio, sr = torchaudio.load(input_wav_path)
    if input_audio.shape[0] != 2:
        raise ValueError("Input audio must have 2 channels.")
    input_audio = input_audio.to(device)
    total_length = input_audio.shape[1]
    instrumentals = torch.zeros_like(input_audio)
    vocals = torch.zeros_like(input_audio)
    cross_fade_length = overlap // 2
    window = torch.hann_window(4096).to(device)
    num_chunks = (total_length - overlap) // (chunk_size - overlap)
    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length - chunk_size + 1, chunk_size - overlap):
            chunk = input_audio[:, i:i + chunk_size]
            chunk_spec = torch.stft(chunk, n_fft=4096, hop_length=1024, window=window, return_complex=True)
            chunk_mag = torch.abs(chunk_spec)
            chunk_phase = torch.angle(chunk_spec)
            chunk_mag = chunk_mag.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_inst_mask, pred_vocal_mask = model(chunk_mag)
            pred_inst_mask = pred_inst_mask.squeeze(0)
            pred_vocal_mask = pred_vocal_mask.squeeze(0)
            pred_inst_mag = chunk_mag.squeeze(0) * pred_inst_mask
            pred_vocal_mag = chunk_mag.squeeze(0) * pred_vocal_mask
            pred_inst_spec = pred_inst_mag * torch.exp(1j * chunk_phase)
            pred_vocal_spec = pred_vocal_mag * torch.exp(1j * chunk_phase)
            inst_chunk = torch.zeros_like(chunk)
            vocal_chunk = torch.zeros_like(chunk)
            for channel in range(2):
                inst_chunk[channel] = torch.istft(
                    pred_inst_spec[channel].unsqueeze(0),
                    n_fft=4096,
                    hop_length=1024,
                    window=window,
                    length=chunk_size,
                    return_complex=False
                ).squeeze(0)
                vocal_chunk[channel] = torch.istft(
                    pred_vocal_spec[channel].unsqueeze(0),
                    n_fft=4096,
                    hop_length=1024,
                    window=window,
                    length=chunk_size,
                    return_complex=False
                ).squeeze(0)
            if i == 0:
                instrumentals[:, i:i + chunk_size] = inst_chunk
                vocals[:, i:i + chunk_size] = vocal_chunk
            else:
                fade_in = torch.linspace(0, 1, cross_fade_length).to(device)
                fade_out = torch.linspace(1, 0, cross_fade_length).to(device)
                inst_chunk[:, :cross_fade_length] *= fade_in
                instrumentals[:, i:i + cross_fade_length] *= fade_out
                instrumentals[:, i:i + cross_fade_length] += inst_chunk[:, :cross_fade_length]
                vocal_chunk[:, :cross_fade_length] *= fade_in
                vocals[:, i:i + cross_fade_length] *= fade_out
                vocals[:, i:i + cross_fade_length] += vocal_chunk[:, :cross_fade_length]
            instrumentals[:, i + cross_fade_length:i + chunk_size] = inst_chunk[:, cross_fade_length:]
            vocals[:, i + cross_fade_length:i + chunk_size] = vocal_chunk[:, cross_fade_length:]
            pbar.update(1)
    instrumentals = torch.clamp(instrumentals, -1.0, 1.0)
    vocals = torch.clamp(vocals, -1.0, 1.0)
    torchaudio.save(output_instrumental_path, instrumentals.cpu(), sr)
    torchaudio.save(output_vocal_path, vocals.cpu(), sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for instrumental separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='augmented_train', help='Path to training dataset')
    parser.add_argument('--preprocess_dir', type=str, default='prep', help='Path to save/load preprocessed data')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint_steps', type=int, default=2000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--segment_length', type=int, default=176400, help='Segment length for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel(in_channels=2, out_channels=2, hidden_channels=252, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters())

    if args.train:
        train_dataset = MUSDBDataset(root_dir=args.data_dir, preprocess_dir=args.preprocess_dir,
                                     segment_length=args.segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=16, pin_memory=False, persistent_workers=True)
        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        train(model, train_dataloader, optimizer, scheduler, loss_fn, device, args.epochs, args.checkpoint_steps, args, checkpoint_path=args.checkpoint_path, window=window)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
