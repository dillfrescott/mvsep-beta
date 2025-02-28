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
from neuralop.models import FNO
import math
import glob
from torch.utils.checkpoint import checkpoint
import random

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=84, n_modes=(86, 86), T=1000, beta_start=1e-4, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.hidden_channels = hidden_channels

        self.projection = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=1)

        self.operator = FNO(n_modes=n_modes, hidden_channels=hidden_channels,
                            in_channels=hidden_channels, out_channels=hidden_channels)

        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self.linear_beta_schedule(timesteps=T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps):
        return torch.linspace(self.beta_start, self.beta_end, timesteps)

    def time_embedding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc  # Return shape [B, channels]

    def forward(self, x, t):
        # x: shape (B, in_channels, H, W)
        # t: shape (B, )
        time_emb = self.time_embedding(t, self.hidden_channels)  # Use self.hidden_channels
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1) # Now [B, hidden_channels, 1, 1]
        time_emb = time_emb.expand(-1, -1, x.shape[2], x.shape[3]) # Now [B, hidden_channels, H, W]

        x = torch.cat((x, time_emb), dim=1) #concat along channel:  [B, in_channels + hidden_channels, H, W]
        x = self.projection(x) # [B, hidden_channels, H, W]

        x = checkpoint(self.operator, x, use_reentrant=False) # [B, hidden_channels, H, W]
        noise = self.mask_predictor(x)  # [B, out_channels, H, W]

        return noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.T)), desc='sampling loop time step', total=self.T, leave=False):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
        return img

    @torch.no_grad()
    def sample(self, shape):
        return self.p_sample_loop(shape)

def loss_fn_diffusion(model, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = model.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = model(x_noisy, t)
    loss = F.mse_loss(noise, predicted_noise)
    return loss

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
            instrumental_mag = torch.from_numpy(data['instrumental_mag'])
            vocal_mag = torch.from_numpy(data['vocal_mag'])
            # Return mixture magnitude, instrumental magnitude, and vocal magnitude
            return mixture_mag, instrumental_mag, vocal_mag
        else:
            track_path = self.tracks[idx]
            _, _, instrumental_mag, _, vocal_mag, _, mixture_mag, _, _ = self._process_track(track_path)
            return torch.from_numpy(mixture_mag), torch.from_numpy(instrumental_mag), torch.from_numpy(vocal_mag)

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
            # Get mixture magnitude and phase from the dataset
            mixture_mag, instrumental_mag, vocal_mag = batch  # Adjust dataset to return mixture_mag

            # Use mixture magnitude as input
            combined_mags = mixture_mag.to(device)  # Shape [B, 2, H, W]

            optimizer.zero_grad()

            # Sample time steps
            t = torch.randint(0, model.T, (combined_mags.shape[0],), device=device).long()
            loss = loss_fn(model, combined_mags, t)  # Use diffusion loss

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
              chunk_size=176400, overlap=44100, device='cpu'):
    # Load model checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval()
    model.to(device)

    # Load input audio
    input_audio, sr = torchaudio.load(input_wav_path)
    if input_audio.shape[0] != 2:
        raise ValueError("Input audio must have 2 channels.")
    input_audio = input_audio.to(device)
    total_length = input_audio.shape[1]

    # Prepare for STFT and ISTFT
    n_fft = 4096
    hop_length = 1024
    window = torch.hann_window(n_fft).to(device)
    num_chunks = (total_length - overlap + (chunk_size - overlap) - 1) // (chunk_size - overlap)

    # Lists to store instrumental and vocal audio chunks
    instrumentals = []
    vocals = []

    with torch.no_grad():
        for i in tqdm(range(num_chunks), desc="Processing audio", total=num_chunks):
            start = i * (chunk_size - overlap)
            end = min(start + chunk_size, total_length)
            current_chunk_size = end - start

            chunk = input_audio[:, start:end]

            # STFT for the chunk
            chunk_spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            chunk_mag = torch.abs(chunk_spec).unsqueeze(0)
            chunk_phase = torch.angle(chunk_spec)

            # Sample from the diffusion model
            sampled_mags = model.sample(chunk_mag.shape).squeeze(0)  # [4, H, W]
            pred_inst_mag, pred_vocal_mag = torch.split(sampled_mags, sampled_mags.shape[0] // 2, dim=0)

            # Reconstruct complex spectrogram using predicted magnitudes and original phase
            pred_inst_spec = pred_inst_mag * torch.exp(1j * chunk_phase)
            pred_vocal_spec = pred_vocal_mag * torch.exp(1j * chunk_phase)

            # Convert back to time domain (for each channel)
            inst_chunk = torch.stack([
                torch.istft(pred_inst_spec[ch], n_fft=n_fft, hop_length=hop_length, window=window, length=current_chunk_size)
                for ch in range(pred_inst_spec.shape[0])
            ], dim=0)
            vocal_chunk = torch.stack([
                torch.istft(pred_vocal_spec[ch], n_fft=n_fft, hop_length=hop_length, window=window, length=current_chunk_size)
                for ch in range(pred_vocal_spec.shape[0])
            ], dim=0)

            instrumentals.append(inst_chunk)
            vocals.append(vocal_chunk)

    # Overlap-add the chunks
    output_instrumental = torch.zeros_like(input_audio)
    output_vocal = torch.zeros_like(input_audio)

    for i, (inst_chunk, vocal_chunk) in enumerate(zip(instrumentals, vocals)):
        start = i * (chunk_size - overlap)
        end = start + inst_chunk.shape[1]  # Use the actual chunk length

        if i == 0:  # First chunk
            output_instrumental[:, start:end] = inst_chunk
            output_vocal[:, start:end] = vocal_chunk
        else:
            # Cross-fade
            cross_fade_length = overlap // 2
            fade_in = torch.linspace(0, 1, cross_fade_length, device=device)
            fade_out = torch.linspace(1, 0, cross_fade_length, device=device)

            # Apply cross-fade to overlapping region in the chunks
            inst_chunk[:, :cross_fade_length] *= fade_in
            vocal_chunk[:, :cross_fade_length] *= fade_in

            # Apply cross-fade to overlapping region in the output
            output_instrumental[:, start:start + cross_fade_length] *= fade_out
            output_vocal[:, start:start + cross_fade_length] *= fade_out

            # Add the cross-faded regions
            output_instrumental[:, start:start + cross_fade_length] += inst_chunk[:, :cross_fade_length]
            output_vocal[:, start:start + cross_fade_length] += vocal_chunk[:, :cross_fade_length]

            # Add non-overlapping region
            output_instrumental[:, start + cross_fade_length:end] = inst_chunk[:, cross_fade_length:]
            output_vocal[:, start + cross_fade_length:end] = vocal_chunk[:, cross_fade_length:]

    # Apply global fade-in and fade-out to smooth edges
    fade_length = 4410  # 0.1 seconds at 44.1kHz
    fade_in = torch.linspace(0, 1, fade_length, device=device)
    fade_out = torch.linspace(1, 0, fade_length, device=device)

    # Apply to instrumental
    output_instrumental[:, :fade_length] *= fade_in
    output_instrumental[:, -fade_length:] *= fade_out

    # Apply to vocal
    output_vocal[:, :fade_length] *= fade_in
    output_vocal[:, -fade_length:] *= fade_out

    # Clamp and save
    output_instrumental = torch.clamp(output_instrumental, -1.0, 1.0)
    output_vocal = torch.clamp(output_vocal, -1.0, 1.0)
    torchaudio.save(output_instrumental_path, output_instrumental.cpu(), sr)
    torchaudio.save(output_vocal_path, output_vocal.cpu(), sr)

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
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = DiffusionModel(in_channels=2, out_channels=2, hidden_channels=84, n_modes=(86, 86))
    optimizer = torch.optim.Adam(model.parameters())

    if args.train:
        train_dataset = MUSDBDataset(root_dir=args.data_dir, preprocess_dir=args.preprocess_dir,
                                     segment_length=args.segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=16, pin_memory=False, persistent_workers=True)
        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        train(model, train_dataloader, optimizer, scheduler, loss_fn_diffusion, device, args.epochs, args.checkpoint_steps, args, checkpoint_path=args.checkpoint_path, window=window)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
