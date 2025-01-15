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
from neuralop.models import FNO, TFNO
import math
import glob
from torch.utils.checkpoint import checkpoint

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        att = self.conv(x)
        att = self.sigmoid(att)
        return x * att

class DynamicAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x

        # Apply initial convolution
        x = self.conv1(x)

        # Dynamic attention mechanism
        attention_map = self.attention(x)
        attended_features = x * attention_map

        # Final convolution
        output = self.final_conv(attended_features)

        # Add residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        output = output + residual

        return output

class NeuralOperatorModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=128, n_modes=(16, 16)):
        super(NeuralOperatorModel, self).__init__()

        # Projection layer to match the input dimension to hidden_channels
        self.projection = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # Dynamic attention with residual connections
        self.attention = DynamicAttention(hidden_channels, hidden_channels)

        # Spatial attention to focus on important regions
        self.spatial_attention = SpatialAttention(hidden_channels)

        # Fourier Neural Operator (FNO) for global feature extraction
        self.operator = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=hidden_channels, out_channels=hidden_channels)

        # Final layer to predict the mask
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Sigmoid()  # Ensure the mask is in the range [0, 1]
        )

    def forward(self, x):
        # Project input to hidden_channels dimension
        x = self.projection(x)  # (batch, hidden_channels, height, width)

        # Apply dynamic attention with checkpointing
        x = checkpoint(self.attention, x, use_reentrant=False)

        # Apply spatial attention with checkpointing
        x = checkpoint(self.spatial_attention, x, use_reentrant=False)

        # Pass through the Fourier Neural Operator
        x = self.operator(x)  # (batch, hidden_channels, height, width)

        # Predict the mask
        mask = self.mask_predictor(x)  # (batch, out_channels, height, width)

        # Split the mask into instrumental and vocal masks
        vocal_mask, inst_mask = torch.split(mask, 1, dim=1)

        return inst_mask, vocal_mask

def loss_fn(pred_inst_mask, pred_vocal_mask, target_inst_mag, target_vocal_mag, mixture_mag, mixture_phase, window, n_fft, hop_length):
    # Ensure the input tensors have the correct shape
    pred_inst_mask = pred_inst_mask.squeeze(0)
    pred_vocal_mask = pred_vocal_mask.squeeze(0)
    target_inst_mag = target_inst_mag.squeeze(0)
    target_vocal_mag = target_vocal_mag.squeeze(0)
    mixture_mag = mixture_mag.squeeze(0)
    mixture_phase = mixture_phase.squeeze(0)

    # Apply the masks to the mixture magnitude
    pred_inst_mag = mixture_mag * pred_inst_mask
    pred_vocal_mag = mixture_mag * pred_vocal_mask

    # Reconstruct time-domain signals using the ground truth phase
    pred_inst_spec = pred_inst_mag * torch.exp(1j * mixture_phase)
    pred_vocal_spec = pred_vocal_mag * torch.exp(1j * mixture_phase)

    # Process each channel separately
    pred_inst_audio = []
    pred_vocal_audio = []
    target_inst_audio = []
    target_vocal_audio = []

    for channel in range(pred_inst_spec.shape[0]):  # Iterate over channels
        # Reconstruct audio for each channel
        pred_inst_audio_ch = torch.istft(pred_inst_spec[channel], n_fft=n_fft, hop_length=hop_length, window=window)
        pred_vocal_audio_ch = torch.istft(pred_vocal_spec[channel], n_fft=n_fft, hop_length=hop_length, window=window)
        target_inst_audio_ch = torch.istft(target_inst_mag[channel] * torch.exp(1j * mixture_phase[channel]), n_fft=n_fft, hop_length=hop_length, window=window)
        target_vocal_audio_ch = torch.istft(target_vocal_mag[channel] * torch.exp(1j * mixture_phase[channel]), n_fft=n_fft, hop_length=hop_length, window=window)

        pred_inst_audio.append(pred_inst_audio_ch)
        pred_vocal_audio.append(pred_vocal_audio_ch)
        target_inst_audio.append(target_inst_audio_ch)
        target_vocal_audio.append(target_vocal_audio_ch)

    # Stack the channels back together
    pred_inst_audio = torch.stack(pred_inst_audio, dim=0)
    pred_vocal_audio = torch.stack(pred_vocal_audio, dim=0)
    target_inst_audio = torch.stack(target_inst_audio, dim=0)
    target_vocal_audio = torch.stack(target_vocal_audio, dim=0)

    # Compute loss in the time domain
    inst_loss = F.l1_loss(pred_inst_audio, target_inst_audio)
    vocal_loss = F.l1_loss(pred_vocal_audio, target_vocal_audio)

    return inst_loss + vocal_loss

# Custom Dataset class
class MUSDBDataset(Dataset):
    def __init__(self, root_dir, preprocess_dir=None, sample_rate=44100, segment_length=264600, n_fft=4096, hop_length=1024, segment=True):
        self.root_dir = root_dir
        self.preprocess_dir = preprocess_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment = segment
        self.tracks = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]
        self.window = torch.hann_window(n_fft)

        # Preprocess data if preprocess_dir is provided
        if self.preprocess_dir:
            self.preprocess_data()

    def preprocess_data(self):
        os.makedirs(self.preprocess_dir, exist_ok=True)
        for idx, track_path in enumerate(tqdm(self.tracks, desc="Preprocessing data")):
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            if not os.path.exists(preprocess_path):
                mixture_mag, mixture_phase, instrumental_mag, instrumental_phase, vocal_mag, vocal_phase, _, _, _ = self._process_track(track_path)
                np.savez(preprocess_path, mixture_mag=mixture_mag, mixture_phase=mixture_phase,
                         instrumental_mag=instrumental_mag, instrumental_phase=instrumental_phase,
                         vocal_mag=vocal_mag, vocal_phase=vocal_phase)

    def _process_track(self, track_path):
        instrumental, _ = torchaudio.load(os.path.join(track_path, 'other.wav'))
        vocal, _ = torchaudio.load(os.path.join(track_path, 'vocals.wav'))

        # Ensure both signals have the same number of channels
        if instrumental.shape[0] != 2 or vocal.shape[0] != 2:
            raise ValueError("Audio files must have 2 channels.")

        # Ensure all signals have the same length
        min_length = min(instrumental.shape[1], vocal.shape[1])
        instrumental = instrumental[:, :min_length]
        vocal = vocal[:, :min_length]

        # Create the mixture by summing instrumental and vocal
        mixture = instrumental + vocal

        # Convert to spectrograms with Hann window
        mixture_spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        instrumental_spec = torch.stft(instrumental, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        vocal_spec = torch.stft(vocal, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)

        # Convert to magnitude and phase spectrograms
        mixture_mag = torch.abs(mixture_spec)
        mixture_phase = torch.angle(mixture_spec)
        instrumental_mag = torch.abs(instrumental_spec)
        instrumental_phase = torch.angle(instrumental_spec)
        vocal_mag = torch.abs(vocal_spec)
        vocal_phase = torch.angle(vocal_spec)

        # Optionally segment the signals
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
                mixture_mag = F.pad(mixture_mag, (0, self.segment_length // self.hop_length - mixture_mag.shape[2]))
                mixture_phase = F.pad(mixture_phase, (0, self.segment_length // self.hop_length - mixture_phase.shape[2]))
                instrumental_mag = F.pad(instrumental_mag, (0, self.segment_length // self.hop_length - instrumental_mag.shape[2]))
                instrumental_phase = F.pad(instrumental_phase, (0, self.segment_length // self.hop_length - instrumental_phase.shape[2]))
                vocal_mag = F.pad(vocal_mag, (0, self.segment_length // self.hop_length - vocal_mag.shape[2]))
                vocal_phase = F.pad(vocal_phase, (0, self.segment_length // self.hop_length - vocal_phase.shape[2]))

        return mixture_mag.numpy(), mixture_phase.numpy(), instrumental_mag.numpy(), instrumental_phase.numpy(), vocal_mag.numpy(), vocal_phase.numpy(), mixture.numpy(), instrumental.numpy(), vocal.numpy()

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
            return self._process_track(track_path)

def adjust_learning_rate(optimizer, grad_norm, base_lr, scale=1.0, eps=1e-8):
    """
    Adjust the learning rate based on the gradient norm.
    Avoids hard limits by using a smooth scaling factor.
    """
    # Avoid division by zero or very small gradient norms
    grad_norm = max(grad_norm, eps)
    
    # Calculate the new learning rate
    lr = base_lr * scale / grad_norm
    
    # Apply a soft constraint to prevent extreme learning rates
    # This ensures the learning rate doesn't explode or vanish
    lr = base_lr * (1.0 / (1.0 + grad_norm / scale))
    
    # Update the learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, dataloader, optimizer, scheduler, loss_fn, device, epochs, checkpoint_steps, args, checkpoint_path=None, window=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    loss_log = []

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        avg_loss = checkpoint['avg_loss']
        loss_log = checkpoint['loss_log']
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

            # Forward pass
            pred_inst_mask, pred_vocal_mask = model(mixture_mag)

            # Compute loss
            loss = loss_fn(pred_inst_mask, pred_vocal_mask, instrumental_mag, vocal_mag, mixture_mag, mixture_phase, window, args.n_fft, args.hop_length)

            # Backward pass with gradient clipping
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

            # Adjust learning rate based on gradient norm
            adjust_learning_rate(optimizer, grad_norm, base_lr=args.learning_rate)

            optimizer.step()
            scheduler.step()

            # Check for NaN in loss
            if torch.isnan(loss).any():
                raise ValueError("Loss is NaN!")

            # Update metrics
            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            loss_log.append(loss.item())
            step += 1
            progress_bar.update(1)

            # Update progress bar description
            current_lr = optimizer.param_groups[0]['lr']
            desc = f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.8f}"
            progress_bar.set_description(desc)

            # Save checkpoint periodically
            if step % checkpoint_steps == 0:
                checkpoint_filename = f"checkpoint_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'loss_log': loss_log
                }, checkpoint_filename)

                # Keep only the last 3 checkpoints
                checkpoint_files = sorted(glob.glob("checkpoint_step_*.pt"), key=os.path.getmtime)
                if len(checkpoint_files) > 3:
                    for old_checkpoint in checkpoint_files[:-3]:
                        os.remove(old_checkpoint)

    # Save final loss log
    torch.save({'loss_log': loss_log}, 'loss_log.pt')
    progress_bar.close()

def inference(model, checkpoint_path, input_wav_path, output_instrumental_path, output_vocal_path,
              chunk_size=88200, overlap=44100, device='cpu', n_fft=4096, hop_length=1024):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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
    window = torch.hann_window(n_fft).to(device)

    num_chunks = (total_length - overlap) // (chunk_size - overlap)
    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length - chunk_size + 1, chunk_size - overlap):
            chunk = input_audio[:, i:i + chunk_size]
            chunk_spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            chunk_mag = torch.abs(chunk_spec)
            chunk_phase = torch.angle(chunk_spec)

            # Ensure the input to the model has the correct shape
            chunk_mag = chunk_mag.unsqueeze(0).to(device)  # Add batch dimension: [1, 2, 2049, 87]

            with torch.no_grad():
                pred_inst_mask, pred_vocal_mask = model(chunk_mag)

            # Remove the batch dimension from the masks
            pred_inst_mask = pred_inst_mask.squeeze(0)  # Shape: [2, 2049, 87]
            pred_vocal_mask = pred_vocal_mask.squeeze(0)  # Shape: [2, 2049, 87]

            # Apply the masks to the mixture magnitude
            pred_inst_mag = chunk_mag.squeeze(0) * pred_inst_mask  # Shape: [2, 2049, 87]
            pred_vocal_mag = chunk_mag.squeeze(0) * pred_vocal_mask  # Shape: [2, 2049, 87]

            # Reconstruct the complex spectrograms
            pred_inst_spec = pred_inst_mag * torch.exp(1j * chunk_phase)  # Shape: [2, 2049, 87]
            pred_vocal_spec = pred_vocal_mag * torch.exp(1j * chunk_phase)  # Shape: [2, 2049, 87]

            # Reconstruct the time-domain signals for each channel
            inst_chunk = torch.zeros_like(chunk)
            vocal_chunk = torch.zeros_like(chunk)

            for channel in range(2):  # Process each channel separately
                inst_chunk[channel] = torch.istft(
                    pred_inst_spec[channel].unsqueeze(0),  # Add batch dimension: [1, 2049, 87]
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    length=chunk_size,
                    return_complex=False
                ).squeeze(0)  # Remove batch dimension

                vocal_chunk[channel] = torch.istft(
                    pred_vocal_spec[channel].unsqueeze(0),  # Add batch dimension: [1, 2049, 87]
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    length=chunk_size,
                    return_complex=False
                ).squeeze(0)  # Remove batch dimension

            # Handle cross-fading between chunks
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

    # Clamp the output to avoid clipping
    instrumentals = torch.clamp(instrumentals, -1.0, 1.0)
    vocals = torch.clamp(vocals, -1.0, 1.0)

    # Save the output audio files
    torchaudio.save(output_instrumental_path, instrumentals.cpu(), sr)
    torchaudio.save(output_vocal_path, vocals.cpu(), sr)

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
    parser.add_argument('--n_fft', type=int, default=4096, help='Number of FFT bins for STFT')
    parser.add_argument('--hop_length', type=int, default=1024, help='Hop length for STFT')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the Hann window for STFT
    window = torch.hann_window(args.n_fft).to(device)

    model = NeuralOperatorModel(in_channels=2, out_channels=2, hidden_channels=128, n_modes=(16, 16))
    optimizer = torch.optim.Adam(model.parameters())

    if args.train:
        # Create the dataset with preprocessing if preprocess_dir is provided
        train_dataset = MUSDBDataset(root_dir=args.data_dir, preprocess_dir=args.preprocess_dir,
                                     segment_length=args.segment_length, n_fft=args.n_fft, hop_length=args.hop_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=16, pin_memory=False, persistent_workers=True)

        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # Pass the window to the train function
        train(model, train_dataloader, optimizer, scheduler, loss_fn, device, args.epochs, args.checkpoint_steps, args, checkpoint_path=args.checkpoint_path, window=window)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        model = NeuralOperatorModel(in_channels=2, out_channels=2, hidden_channels=128, n_modes=(16, 16))
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device, n_fft=args.n_fft, hop_length=args.hop_length)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
