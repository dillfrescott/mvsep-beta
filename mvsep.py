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
from prodigyopt import Prodigy
import math

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
        theta = pos.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        sin, cos = theta.sin(), theta.cos()
        x = x.view(-1, self.dim // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
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
        self.rotary_time = RotaryEmbedding3D(self.dim_each, base)
        self.rotary_mag = RotaryEmbedding3D(self.dim_each, base)
        self.rotary_phase = RotaryEmbedding3D(self.dim_each, base)

    def forward(self, x):
        B, C, H, W = x.shape
        x_time, x_mag, x_phase = torch.chunk(x, 3, dim=1)

        pos_time = torch.arange(W, device=x.device).float().view(1, 1, 1, W)
        pos_mag = torch.arange(H, device=x.device).float().view(1, 1, H, 1)
        pos_phase = torch.linspace(-math.pi, math.pi, H, device=x.device).float().view(1, 1, H, 1)

        x_time = x_time.permute(0, 2, 3, 1)
        x_mag = x_mag.permute(0, 2, 3, 1)
        x_phase = x_phase.permute(0, 2, 3, 1)

        x_time_flat = x_time.reshape(-1, self.dim_each)
        x_mag_flat = x_mag.reshape(-1, self.dim_each)
        x_phase_flat = x_phase.reshape(-1, self.dim_each)

        pos_time_flat = pos_time.expand(B, 1, H, W).reshape(-1)
        pos_mag_flat = pos_mag.expand(B, 1, H, W).reshape(-1)
        pos_phase_flat = pos_phase.expand(B, 1, H, W).reshape(-1)

        x_time_rot = self.rotary_time(x_time_flat, pos_time_flat)
        x_mag_rot = self.rotary_mag(x_mag_flat, pos_mag_flat)
        x_phase_rot = self.rotary_phase(x_phase_flat, pos_phase_flat)

        x_time_rot = x_time_rot.view(B, H, W, self.dim_each)
        x_mag_rot = x_mag_rot.view(B, H, W, self.dim_each)
        x_phase_rot = x_phase_rot.view(B, H, W, self.dim_each)

        x_out = torch.cat([x_time_rot, x_mag_rot, x_phase_rot], dim=-1)
        x_out = x_out.permute(0, 3, 1, 2)
        return x_out

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=128, num_layers=1):
        super(NeuralModel, self).__init__()
        self.projection = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.rotary_pos_emb = RotaryPositionalEmbedding3D(hidden_channels)
        self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels,
                            num_layers=num_layers, batch_first=True)
        self.mag_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels=4, kernel_size=1),
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.rotary_pos_emb(x)
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, F, T, C)
        x = x.reshape(B * F, T, C)
        x, _ = self.lstm(x)
        x = x.reshape(B, F, T, C)
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, F, T)
        magnitudes = self.mag_predictor(x)
        inst_mag, vocal_mag = torch.split(magnitudes, 2, dim=1)
        return inst_mag, vocal_mag

def loss_fn(pred_inst_mag, pred_vocal_mag,
            target_inst_mag, target_vocal_mag,
            mixture_mag, mixture_phase,
            window, n_fft, hop_length,
            consistency_weight=0.1, eps=1e-8):

    def softclip(x, alpha=0.1):
        return F.softplus(x, beta=1 / alpha)

    pred_inst_mag = softclip(pred_inst_mag)
    pred_vocal_mag = softclip(pred_vocal_mag)

    eps_tensor = torch.full_like(pred_inst_mag, eps)

    log_pred_inst_mag = torch.log1p(pred_inst_mag + eps_tensor)
    log_pred_vocal_mag = torch.log1p(pred_vocal_mag + eps_tensor)
    log_target_inst_mag = torch.log1p(target_inst_mag + eps_tensor)
    log_target_vocal_mag = torch.log1p(target_vocal_mag + eps_tensor)

    inst_loss = F.mse_loss(log_pred_inst_mag.view(-1), log_target_inst_mag.view(-1))
    vocal_loss = F.mse_loss(log_pred_vocal_mag.view(-1), log_target_vocal_mag.view(-1))

    def make_complex(mag, phase):
        return mag * torch.exp(1j * phase)

    pred_inst_spec = make_complex(pred_inst_mag, mixture_phase)
    pred_vocal_spec = make_complex(pred_vocal_mag, mixture_phase)
    mixture_spec = make_complex(mixture_mag, mixture_phase)

    def istft_channels(spec):
        if spec.ndim == 3:
            return torch.istft(spec, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=False)
        elif spec.ndim == 4:
            outputs = []
            for ch in range(spec.shape[1]):
                spec_ch = spec[:, ch, :, :]
                outputs.append(torch.istft(spec_ch, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=False))
            return torch.stack(outputs, dim=1)

    pred_inst_audio = istft_channels(pred_inst_spec)
    pred_vocal_audio = istft_channels(pred_vocal_spec)
    mixture_audio = istft_channels(mixture_spec)

    reconstructed_mix = pred_inst_audio + pred_vocal_audio
    consistency_loss = F.mse_loss(reconstructed_mix, mixture_audio)

    total_loss = (inst_loss + vocal_loss) / 2 + consistency_weight * consistency_loss
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
                try:
                    (mixture_mag, mixture_phase, instrumental_mag, instrumental_phase,
                     vocal_mag, vocal_phase, _, _, _) = self._process_track(track_path)
                    np.savez(preprocess_path, mixture_mag=mixture_mag, mixture_phase=mixture_phase,
                         instrumental_mag=instrumental_mag, instrumental_phase=instrumental_phase,
                         vocal_mag=vocal_mag, vocal_phase=vocal_phase)
                except ValueError as e:
                    print(f"Skipping track {track_path} due to error: {e}")
                    continue

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

        target_frames = self.segment_length // self.hop_length + (1 if self.segment_length % self.hop_length != 0 else 0)

        if self.segment and self.segment_length:
            if mixture_mag.shape[2] >= target_frames:
                start = torch.randint(0, mixture_mag.shape[2] - target_frames + 1, (1,)).item()
                mixture_mag = mixture_mag[:, :, start:start + target_frames]
                mixture_phase = mixture_phase[:, :, start:start + target_frames]
                instrumental_mag = instrumental_mag[:, :, start:start + target_frames]
                instrumental_phase = instrumental_phase[:, :, start:start + target_frames]
                vocal_mag = vocal_mag[:, :, start:start + target_frames]
                vocal_phase = vocal_phase[:, :, start:start + target_frames]
            else:
                pad_amount = target_frames - mixture_mag.shape[2]
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

def train(model, dataloader, optimizer, scheduler, loss_fn, device, epochs, checkpoint_steps, args, consistency_weight, checkpoint_path=None, window=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    checkpoint_files = []

    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        step = checkpoint_data['step']
        avg_loss = checkpoint_data['avg_loss']
        print(f"Resuming training from step {step} with average loss {avg_loss:.4f}")

    for epoch in range(epochs):
        model.train()
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as t:
            for batch in t:
                mixture_mag, mixture_phase, instrumental_mag, _, vocal_mag, _ = batch
                mixture_mag = mixture_mag.to(device)
                mixture_phase = mixture_phase.to(device)
                instrumental_mag = instrumental_mag.to(device)
                vocal_mag = vocal_mag.to(device)

                optimizer.zero_grad()
                pred_inst_mag, pred_vocal_mag = model(mixture_mag)

                loss = loss_fn(pred_inst_mag, pred_vocal_mag, instrumental_mag, vocal_mag, mixture_mag, mixture_phase, window, 4096, 1024, consistency_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if torch.isnan(loss).any():
                    raise ValueError("Loss is NaN!")

                avg_loss = (avg_loss * step + loss.item()) / (step + 1)
                step += 1

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
                        os.remove(oldest_checkpoint)

                t.set_postfix(avg_loss=avg_loss, current_loss=loss.item())

def inference(model, checkpoint_path, input_wav_path, output_instrumental_path, output_vocal_path,
              chunk_size=88200, overlap=44100, device='cpu'):

    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
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
                pred_inst_mag, pred_vocal_mag = model(chunk_mag)
            pred_inst_mag = pred_inst_mag.squeeze(0)
            pred_vocal_mag = pred_vocal_mag.squeeze(0)

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
                instrumentals[:, i:i + cross_fade_length] += inst_chunk[:,:cross_fade_length]

                vocal_chunk[:, :cross_fade_length] *= fade_in
                vocals[:, i:i + cross_fade_length] *= fade_out
                vocals[:, i:i + cross_fade_length] += vocal_chunk[:, :cross_fade_length]

                instrumentals[:,i + cross_fade_length:i + chunk_size] = inst_chunk[:, cross_fade_length:]
                vocals[:, i + cross_fade_length:i + chunk_size] = vocal_chunk[:,cross_fade_length:]

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
    parser.add_argument('--segment_length', type=int, default=176400, help='Segment length for training')  #176400
    parser.add_argument('--consistency_weight', type=float, default=0.1, help='Weight for the consistency loss')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--hidden_channels', type=int, default=126, help='Number of hidden channels in the model')
    parser.add_argument('--lr', type=float, default=1.0, help='Initial learning rate for Prodigy')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel(in_channels=2, out_channels=2, hidden_channels=args.hidden_channels, num_layers=4)
    optimizer = Prodigy(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    if args.train:
        train_dataset = MUSDBDataset(root_dir=args.data_dir, preprocess_dir=args.preprocess_dir,
                                      segment_length=args.segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        train(model, train_dataloader, optimizer, scheduler, loss_fn, device, args.epochs, args.checkpoint_steps, args, args.consistency_weight, checkpoint_path=args.checkpoint_path, window=window)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
