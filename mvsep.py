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

class MultiScaleDilatedConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_dil2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.conv_dil4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.conv_dil8 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        x2 = self.conv_dil2(x)
        x4 = self.conv_dil4(x)
        x8 = self.conv_dil8(x)
        x_cat = torch.cat([x2, x4, x8], dim=1)
        return self.output_conv(x_cat)

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=2, fno_modes=(16, 16)):
        super(NeuralModel, self).__init__()

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

        self.center_multiscale = MultiScaleDilatedConv(hidden_channels)

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
        x = self.fno1(x)
        x = self.center_multiscale(x)
        x = self.fno2(x)
        logits = self.final_proj(x)
        mask = self.final_activation(logits)
        return mask, logits

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
            pred_logits, # Add logits here
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

    pred_vocal_mask = torch.clamp(pred_vocal_mask, 0.0, 1.0)
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

    logit_penalty = torch.mean(pred_logits**2)

    total_loss = log_wmse_loss + l2_loss + logit_penalty

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
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                step = checkpoint_data.get('step', 0)
                avg_loss = checkpoint_data.get('avg_loss', 0.0)
                if 'optimizer_state_dict' in checkpoint_data:
                     try:
                        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                     except Exception as e:
                        print(f"Warning: Could not load optimizer state from checkpoint: {e}. Optimizer state will be reinitialized.")
                else:
                    print("Warning: Optimizer state not found in checkpoint. Optimizer state will be reinitialized.")

            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {e}. Starting training from scratch.")
                step = 0
                avg_loss = 0.0
        else:
             print(f"Checkpoint path {checkpoint_path} not found. Starting training from scratch.")
             step = 0
             avg_loss = 0.0

    total_steps_expected = epochs * len(dataloader)
    progress_bar = tqdm(total=total_steps_expected, initial=step)
    model.train()
    start_epoch = (step // len(dataloader)) if len(dataloader) > 0 else 0
    progress_bar.set_description(f"Epoch {start_epoch+1}/{epochs} - Starting...")

    for epoch in range(start_epoch, epochs):
         for batch in dataloader:
            mixture_mag, mixture_phase, vocal_mag, instrumental_mag = batch
            mixture_mag = mixture_mag.to(device)
            mixture_phase = mixture_phase.to(device)
            vocal_mag = vocal_mag.to(device)
            instrumental_mag = instrumental_mag.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred_vocal_mask, pred_logits = model(mixture_mag)

            loss = loss_fn(pred_vocal_mask,
                           pred_logits,
                           vocal_mag,
                           instrumental_mag,
                           mixture_mag,
                           mixture_phase,
                           window,
                           n_fft=4096,
                           hop_length=1024,
                           segment_length=args.segment_length,
                           sample_rate=44100)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN or Inf loss detected at step {step}, skipping batch.")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            adjust_learning_rate(optimizer, grad_norm, base_lr=args.learning_rate)
            optimizer.step()

            current_loss = loss.item()
            if avg_loss == 0.0 and step == start_epoch * len(dataloader):
                 avg_loss = current_loss
            else:
                 avg_loss = 0.98 * avg_loss + 0.02 * current_loss

            step += 1
            progress_bar.update(1)

            current_lr = optimizer.param_groups[0]['lr']
            desc = f"Epoch {epoch+1}/{epochs} - Loss: {current_loss:.4f} - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.8f}"
            progress_bar.set_description(desc)

            if step % checkpoint_steps == 0:
                checkpoint_filename = f"checkpoint_step_{step}.pt"
                save_dict = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'avg_loss': avg_loss,
                 }
                try:
                     save_dict['optimizer_state_dict'] = optimizer.state_dict()
                except Exception as e:
                     print(f"Could not get optimizer state dict for checkpoint: {e}")

                torch.save(save_dict, checkpoint_filename)

                checkpoint_files.append(checkpoint_filename)
                if len(checkpoint_files) > 3:
                    oldest_checkpoint = checkpoint_files.pop(0)
                    if os.path.exists(oldest_checkpoint):
                        try:
                            os.remove(oldest_checkpoint)
                        except OSError as e:
                            print(f"Error removing old checkpoint {oldest_checkpoint}: {e}")

    progress_bar.close()

def inference(model,
              checkpoint_path,
              input_wav_path,
              output_instrumental_path,
              output_vocal_path,
              iterations=3,
              chunk_size=88200,
              overlap=44100,
              device='cpu'):

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model.to(device)

    # Load audio
    audio, sr = torchaudio.load(input_wav_path)
    if sr != 44100:
        raise ValueError(f"Input audio must be 44100Hz, got {sr}Hz.")
    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] != 2:
        raise ValueError("Input audio must be mono or stereo.")

    audio = audio.to(device)
    total_len = audio.shape[1]

    # Prepare outputs
    final_vocals = torch.zeros_like(audio)
    final_inst = torch.zeros_like(audio)

    # STFT parameters
    n_fft = 4096
    hop_length = 1024
    window = torch.hann_window(n_fft).to(device)

    # Overlap-add parameters
    step = max(1, chunk_size - overlap)
    cross = overlap // 2
    num_chunks = math.ceil(max(0, total_len - overlap) / step)

    with tqdm(total=num_chunks, desc=f"Processing {os.path.basename(input_wav_path)}") as pbar:
        for start in range(0, total_len, step):
            end = min(start + chunk_size, total_len)
            chunk = audio[:, start:end]
            L = chunk.shape[1]

            # Pad if too short
            if L < n_fft:
                if start == 0:
                    pad = n_fft - L
                    chunk = F.pad(chunk, (0, pad))
                    L = chunk.shape[1]
                else:
                    pbar.update(1)
                    continue

            # Iterative feedback in time domain
            current = chunk.clone()
            with torch.no_grad():
                for _ in range(iterations):
                    spec = torch.stft(current,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      window=window,
                                      return_complex=True)
                    mag = torch.abs(spec)
                    phase = torch.angle(spec)

                    mask, _ = model(mag.unsqueeze(0))  # [1, C, F, T]
                    mask = mask.squeeze(0)              # [C, F, T]

                    masked = mag * mask * torch.exp(1j * phase)

                    # reconstruct back to time
                    next_wave = torch.zeros_like(current)
                    for ch in range(2):
                        next_wave[ch] = torch.istft(
                            masked[ch].unsqueeze(0),
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            length=current.shape[1],
                            return_complex=False
                        ).squeeze(0)
                    current = next_wave.clamp(-1.0, 1.0)

            vocal_chunk = current
            inst_chunk = chunk - vocal_chunk

            # Overlap-add
            if start == 0:
                # first chunk direct copy
                copy_len = min(L, total_len)
                final_vocals[:, :copy_len] = vocal_chunk[:, :copy_len]
                final_inst[:, :copy_len] = inst_chunk[:, :copy_len]
            else:
                # cross-fade
                fade_in = torch.linspace(0, 1, cross, device=device)
                fade_out = torch.linspace(1, 0, cross, device=device)

                ov_start = start
                ov_end = min(start + cross, total_len)
                act_ov = ov_end - ov_start

                if act_ov > 0:
                    final_vocals[:, ov_start:ov_end] = (
                        final_vocals[:, ov_start:ov_end] * fade_out[:act_ov] +
                        vocal_chunk[:, :act_ov] * fade_in[:act_ov]
                    )
                    final_inst[:, ov_start:ov_end] = (
                        final_inst[:, ov_start:ov_end] * fade_out[:act_ov] +
                        inst_chunk[:, :act_ov] * fade_in[:act_ov]
                    )

                # copy remaining
                rem_start = ov_end
                rem_end = min(start + L, total_len)
                if rem_start < rem_end:
                    offset = rem_start - start
                    final_vocals[:, rem_start:rem_end] = vocal_chunk[:, offset:offset + (rem_end - rem_start)]
                    final_inst[:, rem_start:rem_end] = inst_chunk[:, offset:offset + (rem_end - rem_start)]

            pbar.update(1)

    # save outputs
    torchaudio.save(output_vocal_path, final_vocals.cpu(), sr)
    torchaudio.save(output_instrumental_path, final_inst.cpu(), sr)

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
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for the optimizer')
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
