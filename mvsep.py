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
import math
import glob
from torch.utils.checkpoint import checkpoint

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=884):
        super(NeuralModel, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        )
        
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels
        )
        
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Project input features
        x = self.projection(x)
        
        # Reshape for GRU processing
        batch_size, channels, freq, time = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, F, T, C]
        x = x.reshape(batch_size * freq, time, channels)  # [B*F, T, C]
        
        # Process through GRU - we only need the output, not the hidden state
        x, _ = self.gru(x)  # Unpack the tuple
        
        # Reshape back to original dimensions
        x = x.view(batch_size, freq, time, channels)
        x = x.permute(0, 3, 1, 2)  # [B, C, F, T]
        
        # Predict mask
        vocal_mask = self.mask_predictor(x)
        return vocal_mask.expand(-1, 2, -1, -1)

def loss_fn(pred_vocal_mask,
            target_vocal_mag,
            mixture_mag, mixture_phase,
            window, n_fft, hop_length):

    # Initialize STFT loss
    sad_loss = auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        perceptual_weighting=False,
        sample_rate=44100,
        scale="mel",
        n_bins=128,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Apply the predicted mask to the mixture magnitude
    pred_vocal_mag = mixture_mag * pred_vocal_mask  # [B, 2, F, T]
    pred_instrumental_mag = mixture_mag * (1 - pred_vocal_mask)

    # L1 loss between predicted and target magnitudes
    l1_vocal_loss = F.l1_loss(pred_vocal_mag, target_vocal_mag)

    # Phase-aware loss components
    def make_complex(mag):
        return mag * torch.exp(1j * mixture_phase)

    # Convert to complex spectrograms
    pred_vocal_spec = make_complex(pred_vocal_mag)
    target_vocal_spec = make_complex(target_vocal_mag)

    # ISTFT helper
    def istft_channels(spec):
        batch_size, channels, F, T = spec.shape
        spec_combined = spec.reshape(batch_size * channels, F, T)
        audio = torch.istft(spec_combined, n_fft=n_fft, hop_length=hop_length, window=window)
        audio = audio.reshape(batch_size, channels, -1)
        return audio

    pred_vocal_audio = istft_channels(pred_vocal_spec)
    target_vocal_audio = istft_channels(target_vocal_spec)
    
    # Calculate STFT loss
    sad_loss_value = sad_loss(pred_vocal_audio, target_vocal_audio)
    
    # Combine losses
    total_loss = sad_loss_value + l1_vocal_loss
    
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
                mixture_mag, mixture_phase, vocal_mag, instrumental_mag = self._process_track(track_path)
                np.savez(
                    preprocess_path,
                    mixture_mag=mixture_mag,
                    mixture_phase=mixture_phase,
                    vocal_mag=vocal_mag,
                    instrumental_mag=instrumental_mag
                )

    def _load_and_mix_instrumental(self, track_path):
        # Load all instrumental stems
        drums, _ = torchaudio.load(os.path.join(track_path, 'drums.wav'))
        bass, _ = torchaudio.load(os.path.join(track_path, 'bass.wav'))
        other, _ = torchaudio.load(os.path.join(track_path, 'other.wav'))
        
        # Mix them together to create instrumental
        instrumental = drums + bass + other
        
        # Load vocals
        vocals, _ = torchaudio.load(os.path.join(track_path, 'vocals.wav'))
        
        # Create mixture (should be same as instrumental + vocals)
        mixture = instrumental + vocals
        
        # Ensure all audio is stereo (2 channels)
        if mixture.shape[0] != 2:
            if mixture.shape[0] == 1:
                mixture = mixture.repeat(2, 1)
            else:
                raise ValueError("Audio must be mono or stereo")
                
        if instrumental.shape[0] != 2:
            if instrumental.shape[0] == 1:
                instrumental = instrumental.repeat(2, 1)
                
        if vocals.shape[0] != 2:
            if vocals.shape[0] == 1:
                vocals = vocals.repeat(2, 1)
                
        return mixture, instrumental, vocals

    def _process_track(self, track_path):
        mixture, instrumental, vocals = self._load_and_mix_instrumental(track_path)
        
        # Compute STFTs
        mixture_spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, 
                                 window=self.window, return_complex=True)
        vocal_spec = torch.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length, 
                               window=self.window, return_complex=True)
        instrumental_spec = torch.stft(instrumental, n_fft=self.n_fft, hop_length=self.hop_length,
                                     window=self.window, return_complex=True)

        mixture_mag = torch.abs(mixture_spec)
        mixture_phase = torch.angle(mixture_spec)
        vocal_mag = torch.abs(vocal_spec)
        instrumental_mag = torch.abs(instrumental_spec)

        if self.segment and self.segment_length:
            num_frames = self.segment_length // self.hop_length
            if mixture_mag.shape[2] >= num_frames:
                start = torch.randint(0, mixture_mag.shape[2] - num_frames, (1,))
                mixture_mag = mixture_mag[:, :, start:start + num_frames]
                mixture_phase = mixture_phase[:, :, start:start + num_frames]
                vocal_mag = vocal_mag[:, :, start:start + num_frames]
                instrumental_mag = instrumental_mag[:, :, start:start + num_frames]
            else:
                pad_amount = num_frames - mixture_mag.shape[2]
                mixture_mag = F.pad(mixture_mag, (0, pad_amount))
                mixture_phase = F.pad(mixture_phase, (0, pad_amount))
                vocal_mag = F.pad(vocal_mag, (0, pad_amount))
                instrumental_mag = F.pad(instrumental_mag, (0, pad_amount))

        return (mixture_mag.numpy(), mixture_phase.numpy(),
                vocal_mag.numpy(), instrumental_mag.numpy())

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if self.preprocess_dir:
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            data = np.load(preprocess_path)
            mixture_mag = torch.from_numpy(data['mixture_mag'])
            mixture_phase = torch.from_numpy(data['mixture_phase'])
            vocal_mag = torch.from_numpy(data['vocal_mag'])
            instrumental_mag = torch.from_numpy(data['instrumental_mag'])
            return mixture_mag, mixture_phase, vocal_mag, instrumental_mag
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
            mixture_mag, mixture_phase, vocal_mag, instrumental_mag = batch
            mixture_mag = mixture_mag.to(device)
            mixture_phase = mixture_phase.to(device)
            vocal_mag = vocal_mag.to(device)

            optimizer.zero_grad()
            pred_vocal_mask = model(mixture_mag)
            loss = loss_fn(pred_vocal_mask, vocal_mag, mixture_mag, mixture_phase, window, 4096, 1024)
            
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
    parser.add_argument('--preprocess_dir', type=str, default='prep', help='Path to save/load preprocessed data')
    parser.add_argument('--epochs', type=int, default=1000000, help='Number of epochs to train')
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
