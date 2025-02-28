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
    def __init__(self, in_channels=4, out_channels=4, hidden_channels=84, n_modes=(86, 86), T=1000, beta_start=1e-4, beta_end=0.02, window=None):
        super(DiffusionModel, self).__init__()
        self.window = window
        self.hidden_channels = hidden_channels
        self.projection = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=1)

        self.operator = FNO(n_modes=n_modes, hidden_channels=hidden_channels,
                            in_channels=hidden_channels, out_channels=hidden_channels, use_complex=False)

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
        # Fix: Use unsqueeze and expand instead of repeat
        t = t.unsqueeze(-1)  # Shape: (b, 1)
        t = t.expand(-1, channels // 2)  # Shape: (b, channels//2)
        pos_enc_a = torch.sin(t * inv_freq)
        pos_enc_b = torch.cos(t * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        time_emb = self.time_embedding(t, self.hidden_channels)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        time_emb = time_emb.expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, time_emb), dim=1)
        x = self.projection(x)
        x = checkpoint(self.operator, x, use_reentrant=False)
        noise = self.mask_predictor(x)
        return noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)  # Real-valued noise

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return (sqrt_alphas_cumprod_t * x_start +
                sqrt_one_minus_alphas_cumprod_t * noise)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def p_sample(self, x, t, t_index, mixture_spec_stacked):
        # Concatenate mixture (4 channels) + current noise (8 channels)
        x_input = torch.cat([mixture_spec_stacked, x], dim=1)
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        predicted_noise = self(x_input, t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, mixture_spec_stacked):
        device = next(self.parameters()).device
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.T)), desc='sampling loop time step', total=self.T, leave=False):
            img = self.p_sample(img, torch.full((shape[0],), i, device=device, dtype=torch.long), i, mixture_spec_stacked)
        return img

    @torch.no_grad()
    def sample(self, mixture_spec_stacked):
        b, c, h, w = mixture_spec_stacked.shape
        initial_noise = torch.randn(b, 8, h, w, device=mixture_spec_stacked.device)
        predicted_spectrograms = self.p_sample_loop(initial_noise.shape, mixture_spec_stacked)
        instrumental = predicted_spectrograms[:, :4, :, :]
        vocal = predicted_spectrograms[:, 4:8, :, :]
        return instrumental, vocal

def loss_fn_diffusion(model, x_start_stacked, t, mixture_input, spectrogram_loss):
    noise = torch.randn_like(x_start_stacked)
    x_noisy = model.q_sample(x_start=x_start_stacked, t=t, noise=noise)
    x_input = torch.cat([mixture_input, x_noisy], dim=1)  # Condition on mixture + noisy targets
    predicted_noise = model(x_input, t)
    loss = spectrogram_loss(predicted_noise, noise)
    return loss

class MUSDBDataset(Dataset):
    def __init__(self, root_dir, preprocess_dir=None, sample_rate=44100, segment_length=485100, segment=True):
        super().__init__()
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
                (mixture_spec, instrumental_spec, vocal_spec,
                 mixture, instrumental, vocal) = self._process_track(track_path)

                np.savez(preprocess_path,
                         mixture_real=mixture_spec.real.numpy(),
                         mixture_imag=mixture_spec.imag.numpy(),
                         instrumental_real=instrumental_spec.real.numpy(),
                         instrumental_imag=instrumental_spec.imag.numpy(),
                         vocal_real=vocal_spec.real.numpy(),
                         vocal_imag=vocal_spec.imag.numpy(),
                         mixture_time=mixture.numpy(),
                         instrumental_time=instrumental.numpy(),
                         vocal_time=vocal.numpy())

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

      if self.segment and self.segment_length:
          if mixture_spec.shape[2] >= self.segment_length // self.hop_length:
              start = torch.randint(0, mixture_spec.shape[2] - self.segment_length // self.hop_length, (1,))
              mixture_spec = mixture_spec[:, :, start:start + self.segment_length // self.hop_length]
              instrumental_spec = instrumental_spec[:, :, start:start + self.segment_length // self.hop_length]
              vocal_spec = vocal_spec[:, :, start:start + self.segment_length // self.hop_length]
              start_time = start.item() * self.hop_length
              end_time = start_time + self.segment_length
              mixture = mixture[:, start_time:end_time]
              instrumental = instrumental[:, start_time:end_time]
              vocal = vocal[:, start_time:end_time]

          else:
                pad_amount = self.segment_length // self.hop_length - mixture_spec.shape[2]
                mixture_spec = F.pad(mixture_spec, (0, pad_amount))
                instrumental_spec = F.pad(instrumental_spec, (0, pad_amount))
                vocal_spec = F.pad(vocal_spec, (0, pad_amount))

                pad_amount_time = self.segment_length - mixture.shape[1]
                mixture = F.pad(mixture, (0, pad_amount_time))
                instrumental = F.pad(instrumental, (0, pad_amount_time))
                vocal = F.pad(vocal, (0, pad_amount_time))

      return mixture_spec, instrumental_spec, vocal_spec, mixture, instrumental, vocal

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if self.preprocess_dir:
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            data = np.load(preprocess_path)
            mixture_real = torch.from_numpy(data['mixture_real'])
            mixture_imag = torch.from_numpy(data['mixture_imag'])
            instrumental_real = torch.from_numpy(data['instrumental_real'])
            instrumental_imag = torch.from_numpy(data['instrumental_imag'])
            vocal_real = torch.from_numpy(data['vocal_real'])
            vocal_imag = torch.from_numpy(data['vocal_imag'])

            mixture_time = torch.from_numpy(data['mixture_time'])
            instrumental_time = torch.from_numpy(data['instrumental_time'])
            vocal_time = torch.from_numpy(data['vocal_time'])

            # Reconstruct complex spectrograms for each channel
            mixture_spec = torch.complex(mixture_real, mixture_imag)
            instrumental_spec = torch.complex(instrumental_real, instrumental_imag)
            vocal_spec = torch.complex(vocal_real, vocal_imag)

            return mixture_spec, instrumental_spec, vocal_spec, instrumental_time, vocal_time

        else:
            (mixture_spec, instrumental_spec, vocal_spec, mixture, instrumental, vocal) = self._process_track(self.tracks[idx])
            return mixture_spec, instrumental_spec, vocal_spec, instrumental, vocal

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
    
    spectrogram_loss = nn.MSELoss().to(device)

    for epoch in range(epochs):
        for batch in dataloader:
            mixture_spec, instrumental_spec, vocal_spec, instrumental_time, vocal_time = batch
            mixture_spec = mixture_spec.to(device)
            instrumental_spec = instrumental_spec.to(device)
            vocal_spec = vocal_spec.to(device)
            instrumental_time = instrumental_time.to(device)
            vocal_time = vocal_time.to(device)

            optimizer.zero_grad()
            t = torch.randint(0, model.T, (mixture_spec.shape[0],), device=device).long()

            # Stack along channel dimension (dim=1) for ALL inputs.
            mixture_input = torch.cat([mixture_spec.real, mixture_spec.imag], dim=1)
            instrumental_target = torch.cat([instrumental_spec.real, instrumental_spec.imag], dim=1)
            vocal_target = torch.cat([vocal_spec.real, vocal_spec.imag], dim=1)

            stacked_targets = torch.cat([instrumental_target, vocal_target], dim=1)
            noise = torch.randn_like(stacked_targets)
            x_noisy = model.q_sample(x_start=stacked_targets, t=t, noise=noise)
            x_input = torch.cat([mixture_input, x_noisy], dim=1)

            loss = loss_fn_diffusion(
                model,
                stacked_targets,  # Clean targets (used to generate noisy data)
                t,
                mixture_input,    # Mixture spectrogram (real + imag)
                spectrogram_loss
            )

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
    # Load the checkpoint and prepare the model for inference
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval()
    model.to(device)

    # Load the input audio file
    input_audio, sr = torchaudio.load(input_wav_path)
    if input_audio.shape[0] != 2:
        raise ValueError("Input audio must have 2 channels.")
    input_audio = input_audio.to(device)
    total_length = input_audio.shape[1]

    # STFT parameters
    n_fft = 4096
    hop_length = 1024
    window = torch.hann_window(n_fft).to(device)

    # Calculate the number of chunks
    num_chunks = (total_length - overlap + (chunk_size - overlap) - 1) // (chunk_size - overlap)

    # Lists to store instrumental and vocal chunks
    instrumentals = []
    vocals = []

    # Perform inference on each chunk
    with torch.no_grad():
        for i in tqdm(range(num_chunks), desc="Processing audio", total=num_chunks):
            start = i * (chunk_size - overlap)
            end = min(start + chunk_size, total_length)
            current_chunk_size = end - start

            # Extract the current chunk
            chunk = input_audio[:, start:end]

            # Compute STFT of the current chunk
            chunk_spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            mixture_phase = torch.angle(chunk_spec)
            mixture_mag = torch.abs(chunk_spec)

            # Stack real and imaginary parts along the channel dimension
            chunk_spec_stacked = torch.cat([chunk_spec.real, chunk_spec.imag], dim=0)
            chunk_spec_stacked = chunk_spec_stacked.unsqueeze(0)

            # Predict instrumental and vocal spectrograms
            instrumental_spec, vocal_spec = model.sample(chunk_spec_stacked)
            instrumental_spec = instrumental_spec.squeeze(0)
            vocal_spec = vocal_spec.squeeze(0)

            # Separate real and imaginary parts
            inst_real = instrumental_spec[:2]
            inst_imag = instrumental_spec[2:]
            vocal_real = vocal_spec[:2]
            vocal_imag = vocal_spec[2:]
            
            # Reconstruct complex spectrograms (magnitudes only, for now)
            pred_inst_mag = torch.abs(torch.complex(inst_real, inst_imag))
            pred_vocal_mag = torch.abs(torch.complex(vocal_real, vocal_imag))

            # Normalize predicted magnitudes relative to the mixture magnitude
            magnitude_sum = pred_inst_mag + pred_vocal_mag
            epsilon = 1e-8  # Small constant to prevent division by zero
            pred_inst_mag = pred_inst_mag / (magnitude_sum + epsilon) * mixture_mag
            pred_vocal_mag = pred_vocal_mag / (magnitude_sum + epsilon) * mixture_mag


            # Reconstruct complex spectrograms with mixture phase
            pred_inst_spec = pred_inst_mag * torch.exp(1j * mixture_phase)
            pred_vocal_spec = pred_vocal_mag * torch.exp(1j * mixture_phase)

            # Convert back to time domain
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

    # Combine chunks with cross-fade
    output_instrumental = torch.zeros_like(input_audio)
    output_vocal = torch.zeros_like(input_audio)

    for i, (inst_chunk, vocal_chunk) in enumerate(zip(instrumentals, vocals)):
        start = i * (chunk_size - overlap)
        end = start + inst_chunk.shape[1]

        if i == 0:
            output_instrumental[:, start:end] = inst_chunk
            output_vocal[:, start:end] = vocal_chunk
        else:
            cross_fade_length = overlap // 2
            fade_in = torch.linspace(0, 1, cross_fade_length, device=device)
            fade_out = torch.linspace(1, 0, cross_fade_length, device=device)

            inst_chunk[:, :cross_fade_length] *= fade_in
            vocal_chunk[:, :cross_fade_length] *= fade_in
            output_instrumental[:, start:start + cross_fade_length] *= fade_out
            output_vocal[:, start:start + cross_fade_length] *= fade_out

            output_instrumental[:, start:start + cross_fade_length] += inst_chunk[:, :cross_fade_length]
            output_vocal[:, start:start + cross_fade_length] += vocal_chunk[:, :cross_fade_length]

            output_instrumental[:, start + cross_fade_length:end] = inst_chunk[:, cross_fade_length:]
            output_vocal[:, start + cross_fade_length:end] = vocal_chunk[:, cross_fade_length:]
    
    fade_length = 4410
    fade_in = torch.linspace(0, 1, fade_length, device=device)
    fade_out = torch.linspace(1, 0, fade_length, device=device)
    
    output_instrumental[:, :fade_length] *= fade_in
    output_instrumental[:, -fade_length:] *= fade_out
    output_vocal[:, :fade_length] *= fade_in
    output_vocal[:, -fade_length:] *= fade_out

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
    model = DiffusionModel(in_channels=12, out_channels=8, hidden_channels=84, n_modes=(86, 86))
    optimizer = torch.optim.Adam(model.parameters())

    if args.train:
        train_dataset = MUSDBDataset(root_dir=args.data_dir, preprocess_dir=args.preprocess_dir, segment_length=args.segment_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=16, pin_memory=False, persistent_workers=True)
        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        train(model, train_dataloader, optimizer, scheduler, loss_fn_diffusion, device, args.epochs, args.checkpoint_steps, args, checkpoint_path=args.checkpoint_path, window=window)
    elif args.infer:
        if not args.input_wav:
            raise ValueError("Must provide --input_wav for inference")
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
