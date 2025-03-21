import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from segformer_pytorch import Segformer
from prodigyopt import Prodigy
import auraloss
import numpy as np
import math
import glob
from torch.utils.checkpoint import checkpoint

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=256):
        super(NeuralModel, self).__init__()
        self.projection = nn.Conv2d(in_channels, 3, kernel_size=1)
        
        seg_dims = (32, 64, 160, 256)
        self.segformer1 = Segformer(
            dims=seg_dims,
            heads=(1, 2, 5, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            num_layers=8,
            decoder_dim=256,
            num_classes=hidden_channels
        )
        
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        original_H, original_W = x.shape[-2:]
        x = self.projection(x)  # [B, 3, H, W]
        
        B, C, H, W = x.shape
        # Pad to be divisible by 64
        pad_H = (64 - H % 64) % 64
        pad_W = (64 - W % 64) % 64
        x_padded = F.pad(x, (0, pad_W, 0, pad_H), mode='reflect')
        
        x_seg = self.segformer1(x_padded)
        
        x_upsampled = F.interpolate(x_seg, size=(H + pad_H, W + pad_W), mode='bilinear', align_corners=False)
        x_cropped = x_upsampled[:, :, :H, :W]
        
        # Compute vocal mask and expand it to match the two channels of the input magnitude.
        vocal_mask = self.mask_predictor(x_cropped)  # [B, 1, H, W]
        vocal_mask_expanded = vocal_mask.expand(-1, 2, -1, -1)  # now [B, 2, H, W]
        return vocal_mask_expanded

def loss_fn(pred_vocal_mask,
            target_vocal_mag,
            mixture_mag, mixture_phase,
            window, n_fft, hop_length):

    mrstft = auraloss.freq.MultiResolutionSTFTLoss()
    
    # Apply the predicted mask to the mixture magnitude.
    pred_vocal_mag = mixture_mag * pred_vocal_mask  # [B, 2, F, T]

    # Helper to form a complex spectrogram.
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
    
    # Compute the MRSTFT loss
    loss_vocal = mrstft(pred_vocal_audio, target_vocal_audio)
    total_loss = loss_vocal
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
            if mixture_mag.shape[2] >= self.segment_length // self.hop_length:
                start = torch.randint(0, mixture_mag.shape[2] - self.segment_length // self.hop_length, (1,))
                mixture_mag = mixture_mag[:, :, start:start + self.segment_length // self.hop_length]
                mixture_phase = mixture_phase[:, :, start:start + self.segment_length // self.hop_length]
                vocal_mag = vocal_mag[:, :, start:start + self.segment_length // self.hop_length]
                vocal_phase = vocal_phase[:, :, start:start + self.segment_length // self.hop_length]
            else:
                pad_amount = self.segment_length // self.hop_length - mixture_mag.shape[2]
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
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            step += 1
            progress_bar.update(1)
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
    parser.add_argument('--checkpoint_steps', type=int, default=400, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--segment_length', type=int, default=485100, help='Segment length for training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel(in_channels=2)
    optimizer = Prodigy(model.parameters(), lr=1.0, decouple=False)

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
