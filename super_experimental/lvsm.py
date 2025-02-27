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
from lvsm_pytorch import LVSM
from torch_log_wmse import LogWMSE
import math
import glob
from torch.utils.checkpoint import checkpoint
import shutil

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=512, patch_size=17, n_fft=4096):
        super(NeuralModel, self).__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.max_freq = (n_fft // 2 + 1)
        self.max_freq = max(math.ceil(self.max_freq / patch_size) * patch_size, n_fft // 2 + 1)
        print(f"Using max_freq: {self.max_freq}")

        self.lvsm = LVSM(
            dim=hidden_channels,
            max_image_size=self.max_freq,
            patch_size=patch_size,
            channels=3,
            depth=12,
            dropout_input_ray_prob=0.0
        )

        self.feature_mapper = nn.Conv3d(3, hidden_channels, kernel_size=1)

        self.mask_predictor = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, channels, freq, time = x.shape
        original_time = time  # Store the original time
        pad_freq = self.max_freq - freq  # Calculate padding based on self.max_freq
        pad_time = (self.patch_size - (time % self.patch_size)) % self.patch_size
        if pad_freq > 0 or pad_time > 0:
            x = F.pad(x, (0, pad_time, 0, pad_freq))  # Apply padding
            freq, time = x.shape[2], x.shape[3]
        if channels == 2:
            x = torch.cat([x, torch.zeros(batch_size, 1, freq, time, device=x.device)], dim=1)
            channels = 3

        freq_coords = torch.linspace(0, 1, freq, device=x.device).view(1, freq, 1).expand(batch_size, freq, time)
        time_coords = torch.linspace(0, 1, time, device=x.device).view(1, 1, time).expand(batch_size, freq, time)
        rays = torch.stack([freq_coords, time_coords], dim=1)
        rays = rays.repeat(1, 3, 1, 1)

        x = x.unsqueeze(1)
        rays = rays.unsqueeze(1)

        lvsm_out = checkpoint(self.lvsm, input_images=x, input_rays=rays, target_rays=rays, use_reentrant=False)

        if lvsm_out.dim() == 4:
            lvsm_out = lvsm_out.unsqueeze(2)

        if lvsm_out.size(1) == 1 and lvsm_out.size(2) == 3:
            lvsm_out = lvsm_out.permute(0, 2, 1, 3, 4)

        lvsm_feat = self.feature_mapper(lvsm_out)
        mask = self.mask_predictor(lvsm_feat)
        mask = mask.squeeze(2)
        mask = mask[..., :original_time]
        vocal_mask, inst_mask = torch.split(mask, 1, dim=1)

        return inst_mask, vocal_mask
        
def loss_fn(pred_inst_mask, pred_vocal_mask,
            target_inst_mag, target_vocal_mag,
            mixture_mag, mixture_phase,
            window, n_fft, hop_length, original_length, padded_length):
    pred_inst_mag = mixture_mag * pred_inst_mask
    pred_vocal_mag = mixture_mag * pred_vocal_mask

    def make_complex(mag):
        return mag * torch.exp(1j * mixture_phase)

    pred_inst_spec = make_complex(pred_inst_mag)
    pred_vocal_spec = make_complex(pred_vocal_mag)
    target_inst_spec = make_complex(target_inst_mag)
    target_vocal_spec = make_complex(target_vocal_mag)
    mixture_spec = make_complex(mixture_mag)

    def truncate_freq(spec):
        return spec[:, :, :n_fft // 2 + 1, :]

    pred_inst_spec = truncate_freq(pred_inst_spec)
    pred_vocal_spec = truncate_freq(pred_vocal_spec)
    target_inst_spec = truncate_freq(target_inst_spec)
    target_vocal_spec = truncate_freq(target_vocal_spec)
    mixture_spec = truncate_freq(mixture_spec)

    def istft_channels(spec, length):
        batch_size, channels, _, _ = spec.shape
        output = []
        for b in range(batch_size):
            channel_output = []
            for ch in range(channels):
                channel_output.append(
                    torch.istft(spec[b, ch], n_fft=n_fft, hop_length=hop_length, window=window, length=length[b])  # Use length[b]
                )
            output.append(torch.stack(channel_output, dim=0))
        return torch.stack(output, dim=0)

    pred_inst_audio = istft_channels(pred_inst_spec, padded_length)
    pred_vocal_audio = istft_channels(pred_vocal_spec, padded_length)
    target_inst_audio = istft_channels(target_inst_spec, padded_length)
    target_vocal_audio = istft_channels(target_vocal_spec, padded_length)
    mixture_audio = istft_channels(mixture_spec, padded_length)

    processed_audio = torch.stack([pred_inst_audio, pred_vocal_audio], dim=1)
    target_audio = torch.stack([target_inst_audio, target_vocal_audio], dim=1)

    log_wmse = LogWMSE(
        audio_length=pred_inst_audio.shape[-1] / 44100,
        sample_rate=44100,
        return_as_loss=True,
        bypass_filter=False
    )
    logwmse_loss = log_wmse(mixture_audio, processed_audio, target_audio)
    total_loss = logwmse_loss

    return total_loss

class MUSDBDataset(Dataset):
    def __init__(self, root_dir, preprocess_dir=None, sample_rate=44100, segment_length=485100, segment=True, n_fft=4096, hop_length=1024, patch_size=17):
        super().__init__()
        self.root_dir = root_dir
        self.preprocess_dir = preprocess_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment = segment
        self.tracks = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]
        self.window = torch.hann_window(self.n_fft)
        self.patch_size = patch_size
        self.max_freq = (self.n_fft // 2 + 1)
        self.max_freq = max(math.ceil(self.max_freq / self.patch_size) * self.patch_size, self.n_fft // 2 + 1)

        if self.preprocess_dir:
            self.preprocess_data()

    def preprocess_data(self):
        os.makedirs(self.preprocess_dir, exist_ok=True)
        for idx, track_path in enumerate(tqdm(self.tracks, desc="Preprocessing data")):
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            if not os.path.exists(preprocess_path):
                (mixture_mag, mixture_phase, instrumental_mag, instrumental_phase,
                 vocal_mag, vocal_phase, original_length, padded_length) = self._process_track(track_path)

                np.savez(preprocess_path, mixture_mag=mixture_mag, mixture_phase=mixture_phase,
                         instrumental_mag=instrumental_mag, instrumental_phase=instrumental_phase,
                         vocal_mag=vocal_mag, vocal_phase=vocal_phase, original_length=original_length,
                         padded_length=padded_length)

    def _process_track(self, track_path):
        instrumental, _ = torchaudio.load(os.path.join(track_path, 'other.wav'))
        vocal, _ = torchaudio.load(os.path.join(track_path, 'vocals.wav'))

        if instrumental.shape[0] != 2 or vocal.shape[0] != 2:
            raise ValueError("Audio files must have 2 channels.")

        min_length = min(instrumental.shape[1], vocal.shape[1])
        instrumental = instrumental[:, :min_length]
        vocal = vocal[:, :min_length]
        mixture = instrumental + vocal

        original_length = mixture.shape[1]

        if self.segment and self.segment_length:
            if original_length >= self.segment_length:
                start = torch.randint(0, original_length - self.segment_length + 1, (1,)).item()
                mixture = mixture[:, start:start + self.segment_length]
                instrumental = instrumental[:, start:start + self.segment_length]
                vocal = vocal[:, start:start + self.segment_length]
            else:
                pad_amount = self.segment_length - original_length
                mixture = F.pad(mixture, (0, pad_amount))
                instrumental = F.pad(instrumental, (0, pad_amount))
                vocal = F.pad(vocal, (0, pad_amount))

        input_length = mixture.shape[1]
        num_frames = (input_length - self.n_fft) // self.hop_length + 1
        target_input_length = (num_frames - 1) * self.hop_length + self.n_fft
        padding_needed = target_input_length - input_length
        if padding_needed > 0:
            mixture = F.pad(mixture, (0, padding_needed))
            instrumental = F.pad(instrumental, (0, padding_needed))
            vocal = F.pad(vocal, (0, padding_needed))

        padded_length = mixture.shape[1]

        mixture_spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        instrumental_spec = torch.stft(instrumental, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        vocal_spec = torch.stft(vocal, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)

        mixture_mag = torch.abs(mixture_spec)
        mixture_phase = torch.angle(mixture_spec)
        instrumental_mag = torch.abs(instrumental_spec)
        instrumental_phase = torch.angle(instrumental_spec)
        vocal_mag = torch.abs(vocal_spec)
        vocal_phase = torch.angle(vocal_spec)

        def pad_freq_dim(tensor):
            current_freq = tensor.size(1)
            pad_amount = self.max_freq - current_freq
            if pad_amount > 0:
                return F.pad(tensor, (0, 0, 0, pad_amount))
            return tensor

        mixture_mag = pad_freq_dim(mixture_mag)
        mixture_phase = pad_freq_dim(mixture_phase)
        instrumental_mag = pad_freq_dim(instrumental_mag)
        instrumental_phase = pad_freq_dim(instrumental_phase)
        vocal_mag = pad_freq_dim(vocal_mag)
        vocal_phase = pad_freq_dim(vocal_phase)

        return (mixture_mag.numpy(), mixture_phase.numpy(),
                instrumental_mag.numpy(), instrumental_phase.numpy(),
                vocal_mag.numpy(), vocal_phase.numpy(),
                original_length, padded_length)

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
            original_length = torch.tensor(data['original_length']).long()
            padded_length = torch.tensor(data['padded_length']).long()
            return (mixture_mag, mixture_phase, instrumental_mag, instrumental_phase, vocal_mag, vocal_phase, original_length, padded_length)
        else:
            track_path = self.tracks[idx]
            return self._process_track(track_path)
            
def adjust_learning_rate(optimizer, grad_norm, base_lr, scale=1.0, eps=1e-8):
    grad_norm = max(grad_norm, eps)
    lr = base_lr * (1.0 / (1.0 + grad_norm / scale))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, dataloader, optimizer, scheduler, loss_fn, device, epochs, checkpoint_steps, args, checkpoint_dir="checkpoints", checkpoint_path=None, window=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)

    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        step = checkpoint_data['step']
        avg_loss = checkpoint_data['avg_loss']
        print(f"Resuming training from step {step} with average loss {avg_loss:.4f}")

    progress_bar = tqdm(total=epochs * len(dataloader))
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            mixture_mag, mixture_phase, instrumental_mag, _, vocal_mag, _, original_length, padded_length = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pred_inst_mask, pred_vocal_mask = model(mixture_mag)

            loss = loss_fn(pred_inst_mask, pred_vocal_mask, instrumental_mag, vocal_mag, mixture_mag, mixture_phase, window, args.n_fft, args.hop_length, original_length, padded_length)
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
                checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_step_{step}.pt")
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss
                }
                torch.save(checkpoint_data, checkpoint_filename)

                checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*_step_*.pt")), key=lambda x: (int(x.split('_')[2]), int(x.split('_')[4].split('.')[0])))
                while len(checkpoints) > 3:
                    os.remove(checkpoints.pop(0))
    progress_bar.close()

def inference(model, checkpoint_path, input_wav_path, output_instrumental_path, output_vocal_path,
              chunk_size=88200, overlap=44100, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
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

            pad_amount = model.max_freq - chunk_mag.shape[1]
            if pad_amount > 0:
                chunk_mag = F.pad(chunk_mag, (0, 0, 0, pad_amount))
                chunk_phase = F.pad(chunk_phase, (0, 0, 0, pad_amount))

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

            # Truncate before iSTFT, similar to the loss_fn
            pred_inst_spec = pred_inst_spec[:, :model.n_fft // 2 + 1, :]
            pred_vocal_spec = pred_vocal_spec[:, :model.n_fft // 2 + 1, :]

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
    parser.add_argument('--data_dir', type=str, default='train', help='Path to training dataset')
    parser.add_argument('--preprocess_dir', type=str, default='prep_new', help='Path to save/load preprocessed data')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint_steps', type=int, default=2000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--segment_length', type=int, default=485100, help='Segment length for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--patch_size', type=int, default=32, help='patch size for lvsm')
    parser.add_argument('--n_fft', type=int, default=4096, help='n_fft for STFT')
    parser.add_argument('--hop_length', type=int, default=1024, help='Hop length for STFT')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(args.n_fft).to(device)
    model = NeuralModel(in_channels=2, out_channels=2, hidden_channels=256, patch_size=args.patch_size, n_fft=args.n_fft)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.train:
        train_dataset = MUSDBDataset(
            root_dir=args.data_dir,
            preprocess_dir=args.preprocess_dir,
            segment_length=args.segment_length,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            patch_size=args.patch_size
        )
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
