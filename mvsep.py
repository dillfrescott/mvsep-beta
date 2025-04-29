import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch_log_wmse import LogWMSE
from neuralop.models import FNO
import numpy as np
import random
import math
import glob
from torch.utils.checkpoint import checkpoint
import warnings

warnings.filterwarnings('ignore')

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
    def __init__(self, in_channels=2, hidden_channels=128, out_channels=2, fno_modes=(16,16)):
        super(NeuralModel, self).__init__()
        
        unet_depth_channels = [hidden_channels, hidden_channels*2, hidden_channels*4]

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        )
        
        self.fno1 = FNO(
            n_modes=fno_modes,
            hidden_channels=hidden_channels,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_layers=1 
        )

        self.down1 = DownBlock(unet_depth_channels[0], unet_depth_channels[1])
        self.down2 = DownBlock(unet_depth_channels[1], unet_depth_channels[2])
        
        self.center_conv = nn.Sequential(
            UNetConvBlock(unet_depth_channels[2], unet_depth_channels[2]),
            UNetConvBlock(unet_depth_channels[2], unet_depth_channels[2]),
        )

        self.up1 = UpBlock(unet_depth_channels[2] + unet_depth_channels[2], unet_depth_channels[1])
        self.up2 = UpBlock(unet_depth_channels[1] + unet_depth_channels[1], unet_depth_channels[0])

        self.fno2 = FNO(
            n_modes=fno_modes,
            hidden_channels=hidden_channels,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_layers=1 
        )
        
        self.final_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.projection(x)
        
        x_fno1_out = self.fno1(x)

        skip1, down1 = self.down1(x_fno1_out)
        skip2, down2 = self.down2(down1)
        
        center = self.center_conv(down2)
        
        up1 = self.up1(center, skip2)
        x_unet_out = self.up2(up1, skip1)

        x = self.fno2(x_unet_out)
        
        pred_vocal_mask = self.final_proj(x)
        final_vocal_mask = self.final_activation(pred_vocal_mask)
        
        return final_vocal_mask

def istft_channels(spec, n_fft, hop_length, window, target_length):
    batch_size, channels, F_dim, T_dim = spec.shape
    spec_combined = spec.reshape(batch_size * channels, F_dim, T_dim)
    if window is not None:
        window = window.to(spec_combined.device)

    audio = torch.istft(
        spec_combined,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window.shape[0] if window is not None else n_fft,
        window=window,
        return_complex=False,
        center=True,
        length=target_length
    )
    audio = audio.reshape(batch_size, channels, -1)
    return audio

def loss_fn(pred_vocal_mask,
            target_vocal_mag,
            target_instrumental_mag,
            mixture_mag,
            mixture_phase,
            window,
            n_fft,
            hop_length,
            segment_length,
            sample_rate):

    device = mixture_mag.device
    audio_length_seconds = segment_length / sample_rate

    log_wmse_calculator = LogWMSE(
        audio_length=audio_length_seconds,
        sample_rate=sample_rate,
        return_as_loss=True,
        bypass_filter=False
    )

    pred_vocal_mag = mixture_mag * pred_vocal_mask
    pred_instrumental_mag = mixture_mag * (1.0 - pred_vocal_mask)

    pred_vocal_mag = torch.clamp(pred_vocal_mag, min=1e-8)
    pred_instrumental_mag = torch.clamp(pred_instrumental_mag, min=1e-8)
    target_vocal_mag = torch.clamp(target_vocal_mag, min=1e-8)
    target_instrumental_mag = torch.clamp(target_instrumental_mag, min=1e-8)
    mixture_mag_clamped = torch.clamp(mixture_mag, min=1e-8)

    mixture_spec = torch.polar(mixture_mag_clamped, mixture_phase)
    pred_vocal_spec = torch.polar(pred_vocal_mag, mixture_phase)
    target_vocal_spec = torch.polar(target_vocal_mag, mixture_phase)
    pred_instrumental_spec = torch.polar(pred_instrumental_mag, mixture_phase)
    target_instrumental_spec = torch.polar(target_instrumental_mag, mixture_phase)

    unprocessed_audio = istft_channels(mixture_spec, n_fft, hop_length, window, segment_length)
    pred_vocal_audio = istft_channels(pred_vocal_spec, n_fft, hop_length, window, segment_length)
    target_vocal_audio = istft_channels(target_vocal_spec, n_fft, hop_length, window, segment_length)
    pred_instrumental_audio = istft_channels(pred_instrumental_spec, n_fft, hop_length, window, segment_length)
    target_instrumental_audio = istft_channels(target_instrumental_spec, n_fft, hop_length, window, segment_length)

    min_len = min(unprocessed_audio.shape[-1], pred_vocal_audio.shape[-1], target_vocal_audio.shape[-1],
                  pred_instrumental_audio.shape[-1], target_instrumental_audio.shape[-1])

    unprocessed_audio = unprocessed_audio[..., :min_len]
    pred_vocal_audio = pred_vocal_audio[..., :min_len]
    target_vocal_audio = target_vocal_audio[..., :min_len]
    pred_instrumental_audio = pred_instrumental_audio[..., :min_len]
    target_instrumental_audio = target_instrumental_audio[..., :min_len]

    processed_audio = torch.stack([pred_vocal_audio, pred_instrumental_audio], dim=2)
    target_audio = torch.stack([target_vocal_audio, target_instrumental_audio], dim=2)

    log_wmse_loss = log_wmse_calculator(unprocessed_audio, processed_audio, target_audio)
    
    mse_loss_calculator = nn.MSELoss()

    l2_loss = mse_loss_calculator(pred_vocal_mag, target_vocal_mag)

    total_loss = log_wmse_loss + l2_loss

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
            loss = loss_fn(pred_vocal_mask,
                                   vocal_mag,
                                   instrumental_mag,
                                   mixture_mag,
                                   mixture_phase,
                                   window,
                                   n_fft=4096,
                                   hop_length=1024,
                                   segment_length=args.segment_length,
                                   sample_rate=44100)
            
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
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate for the optimizer')
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
