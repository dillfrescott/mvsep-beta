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

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super(RotaryPositionEmbedding, self).__init__()
        self.dim = dim

    def forward(self, x, seq_len=None):
        batch_size, seq_len, dim = x.size()
        if dim != self.dim:
            raise ValueError(f"Input dimension {dim} does not match RoPE dimension {self.dim}")

        # Generate position indices
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(-1)

        # Compute the angles for rotation
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) * -(math.log(10000.0) / dim))
        angles = position * div_term

        # Create sin and cos embeddings
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)

        # Apply rotation to the input
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, 0::2] = x[:, :, 0::2] * cos_emb - x[:, :, 1::2] * sin_emb
        x_rotated[:, :, 1::2] = x[:, :, 1::2] * cos_emb + x[:, :, 0::2] * sin_emb

        return x_rotated

class HigherDimensionalProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HigherDimensionalProjection, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class NeuralOperatorModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=128, projected_channels=512, n_modes=(48, 48), factorization=None, rank=0.05):
        super(NeuralOperatorModel, self).__init__()
        # Define the higher-dimensional projection
        self.projection = nn.Sequential(
            nn.Linear(in_channels, projected_channels),  # Project to higher dimensions
            nn.ReLU(),  # Add non-linearity
            nn.Linear(projected_channels, projected_channels)
        )

        # Define the neural operator
        if factorization is None:
            self.operator = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=projected_channels, out_channels=out_channels)
        else:
            self.operator = TFNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=projected_channels, out_channels=out_channels, factorization=factorization, rank=rank)

        # Rotary Position Embedding
        self.rope = RotaryPositionEmbedding(dim=in_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Reshape and apply Rotary Position Embedding
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, height * width, -1)
        x = self.rope(x, seq_len=height * width)
        x = x.reshape(batch_size, height, width, -1)

        # Project into higher-dimensional space
        x = self.projection(x)

        # Reshape for the neural operator
        x = x.permute(0, 3, 1, 2)

        # Apply the neural operator
        x = self.operator(x)

        # Split the output into instrumental and vocal magnitudes
        pred_inst_mag, pred_vocal_mag = torch.split(x, 1, dim=1)
        return pred_inst_mag, pred_vocal_mag

def loss_fn(pred_inst_mag, pred_vocal_mag, target_inst_mag, target_vocal_mag, mixture_phase, window, n_fft, hop_length):
    # Ensure the input tensors have the correct shape
    pred_inst_mag = pred_inst_mag.squeeze(0)
    pred_vocal_mag = pred_vocal_mag.squeeze(0)
    target_inst_mag = target_inst_mag.squeeze(0)
    target_vocal_mag = target_vocal_mag.squeeze(0)
    mixture_phase = mixture_phase.squeeze(0)

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
    """
    lr = base_lr * scale / (grad_norm + eps)
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
            # Unpack the batch
            mixture_mag, mixture_phase, instrumental_mag, _, vocal_mag, _ = batch
            mixture_mag = mixture_mag.to(device)
            mixture_phase = mixture_phase.to(device)
            instrumental_mag = instrumental_mag.to(device)
            vocal_mag = vocal_mag.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_inst_mag, pred_vocal_mag = model(mixture_mag)

            # Compute loss in the time domain
            loss = loss_fn(pred_inst_mag, pred_vocal_mag, instrumental_mag, vocal_mag, mixture_phase, window, args.n_fft, args.hop_length)

            # Backward pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            adjust_learning_rate(optimizer, grad_norm, base_lr=args.learning_rate)

            optimizer.step()
            scheduler.step()

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
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'loss_log': loss_log
                }, f"checkpoint_step_{step}.pt")

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

            chunk_mag = chunk_mag.unsqueeze(0).to(device)

            with torch.no_grad():
                pred_inst_mag, pred_vocal_mag = model(chunk_mag)

            pred_inst_mag = pred_inst_mag.squeeze(0)
            pred_vocal_mag = pred_vocal_mag.squeeze(0)

            pred_inst_spec = pred_inst_mag * torch.exp(1j * chunk_phase)
            pred_vocal_spec = pred_vocal_mag * torch.exp(1j * chunk_phase)

            inst_chunk = torch.istft(pred_inst_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=chunk_size, return_complex=False)
            vocal_chunk = torch.istft(pred_vocal_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=chunk_size, return_complex=False)

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
    parser.add_argument('--factorization', type=str, default=None, help='Factorization type for TFNO (e.g., "tucker")')
    parser.add_argument('--rank', type=float, default=0.05, help='Rank for TFNO factorization')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the Hann window for STFT
    window = torch.hann_window(args.n_fft).to(device)

    model = NeuralOperatorModel(in_channels=2, out_channels=2, hidden_channels=128, n_modes=(48, 48),
                                factorization=args.factorization, rank=args.rank)
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
        model = NeuralOperatorModel(in_channels=2, out_channels=2, hidden_channels=128, n_modes=(48, 48),
                                    factorization=args.factorization, rank=args.rank)
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device, n_fft=args.n_fft, hop_length=args.hop_length)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
