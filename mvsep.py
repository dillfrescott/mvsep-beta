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

class NeuralOperatorModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=4, hidden_channels=64, n_modes=(16, 16), factorization=None, rank=0.05):
        super(NeuralOperatorModel, self).__init__()
        if factorization is None:
            self.operator = FNO(n_modes=n_modes, hidden_channels=hidden_channels,
                                in_channels=in_channels, out_channels=out_channels)
        else:
            self.operator = TFNO(n_modes=n_modes, hidden_channels=hidden_channels,
                                 in_channels=in_channels, out_channels=out_channels,
                                 factorization=factorization, rank=rank)

    def forward(self, x):
        # Pass through the neural operator
        x = self.operator(x)
        # Split the output into magnitude and phase
        pred_inst_mag, pred_vocal_mag, pred_inst_phase, pred_vocal_phase = torch.split(x, 1, dim=1)
        return pred_inst_mag, pred_vocal_mag, pred_inst_phase, pred_vocal_phase

def loss_fn(pred_inst_mag, pred_vocal_mag, pred_inst_phase, pred_vocal_phase, 
            target_inst_mag, target_vocal_mag, target_inst_phase, target_vocal_phase,
            vocal_mask, segment_length, sample_rate=44100, hop_length=1024):
    """
    Compute the loss with a focus on vocal sections.
    Args:
        vocal_mask (torch.Tensor): Binary mask (1 for vocals, 0 for non-vocals).
        segment_length (int): Length of the segment in samples (from the main function).
        sample_rate (int): Sample rate of the audio.
        hop_length (int): Hop length for STFT.
    """
    # Calculate the number of time frames corresponding to the segment length
    segment_frames = segment_length // hop_length

    # Initialize the total loss
    total_loss = 0.0

    # Iterate over smaller segments
    for i in range(0, pred_inst_mag.shape[2], segment_frames):
        # Extract the current segment
        pred_inst_mag_seg = pred_inst_mag[:, :, i:i + segment_frames]
        pred_vocal_mag_seg = pred_vocal_mag[:, :, i:i + segment_frames]
        pred_inst_phase_seg = pred_inst_phase[:, :, i:i + segment_frames]
        pred_vocal_phase_seg = pred_vocal_phase[:, :, i:i + segment_frames]

        target_inst_mag_seg = target_inst_mag[:, :, i:i + segment_frames]
        target_vocal_mag_seg = target_vocal_mag[:, :, i:i + segment_frames]
        target_inst_phase_seg = target_inst_phase[:, :, i:i + segment_frames]
        target_vocal_phase_seg = target_vocal_phase[:, :, i:i + segment_frames]

        vocal_mask_seg = vocal_mask[:, :, i:i + segment_frames]

        # Compute loss for vocal sections
        vocal_mag_loss = torch.mean(torch.abs(pred_inst_mag_seg * vocal_mask_seg - target_inst_mag_seg * vocal_mask_seg))
        vocal_phase_loss = torch.mean(torch.abs(torch.sin((pred_inst_phase_seg - target_inst_phase_seg) / 2)))

        # Compute loss for non-vocal sections (use ground truth directly)
        non_vocal_mag_loss = torch.mean(torch.abs(pred_inst_mag_seg * (1 - vocal_mask_seg) - target_inst_mag_seg * (1 - vocal_mask_seg)))
        non_vocal_phase_loss = torch.mean(torch.abs(torch.sin((pred_inst_phase_seg - target_inst_phase_seg) / 2)))

        # Weighted sum of losses
        segment_loss = vocal_mag_loss + 0.5 * non_vocal_mag_loss + vocal_phase_loss + 0.5 * non_vocal_phase_loss
        total_loss += segment_loss

    # Normalize the total loss by the number of segments
    total_loss /= (pred_inst_mag.shape[2] // segment_frames)
    return total_loss

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

    def _create_vocal_mask(self, vocal_mag, threshold=0.1):
        """
        Create a binary mask for vocal sections based on the magnitude spectrogram.
        Args:
            vocal_mag (torch.Tensor): Magnitude spectrogram of the vocal track.
            threshold (float): Threshold for detecting vocal activity.
        Returns:
            vocal_mask (torch.Tensor): Binary mask (1 for vocals, 0 for non-vocals).
        """
        vocal_mask = (vocal_mag > threshold).float()
        return vocal_mask

    def preprocess_data(self):
        """
        Preprocess all tracks and save the results to the preprocess directory.
        """
        os.makedirs(self.preprocess_dir, exist_ok=True)
        for idx, track_path in enumerate(tqdm(self.tracks, desc="Preprocessing data")):
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            if not os.path.exists(preprocess_path):
                # Process the track and save the results
                mixture_mag, mixture_phase, instrumental_mag, instrumental_phase, vocal_mag, vocal_phase, vocal_mask = self._process_track(track_path)
                np.savez(preprocess_path, 
                         mixture_mag=mixture_mag, mixture_phase=mixture_phase,
                         instrumental_mag=instrumental_mag, instrumental_phase=instrumental_phase,
                         vocal_mag=vocal_mag, vocal_phase=vocal_phase,
                         vocal_mask=vocal_mask)

    def _process_track(self, track_path):
        """
        Process a single track to generate magnitude, phase, and vocal mask.
        """
        # Load audio files
        instrumental, _ = torchaudio.load(os.path.join(track_path, 'other.wav'))
        vocal, _ = torchaudio.load(os.path.join(track_path, 'vocals.wav'))

        # Ensure both signals have the same number of channels and length
        if instrumental.shape[0] != 2 or vocal.shape[0] != 2:
            raise ValueError("Audio files must have 2 channels.")
        min_length = min(instrumental.shape[1], vocal.shape[1])
        instrumental = instrumental[:, :min_length]
        vocal = vocal[:, :min_length]

        # Create the mixture by summing instrumental and vocal
        mixture = instrumental + vocal

        # Compute STFTs
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

        # Create vocal mask
        vocal_mask = self._create_vocal_mask(vocal_mag)

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
                vocal_mask = vocal_mask[:, :, start:start + self.segment_length // self.hop_length]
            else:
                # Pad if the segment is shorter than the desired length
                pad_length = self.segment_length // self.hop_length - mixture_mag.shape[2]
                mixture_mag = F.pad(mixture_mag, (0, pad_length))
                mixture_phase = F.pad(mixture_phase, (0, pad_length))
                instrumental_mag = F.pad(instrumental_mag, (0, pad_length))
                instrumental_phase = F.pad(instrumental_phase, (0, pad_length))
                vocal_mag = F.pad(vocal_mag, (0, pad_length))
                vocal_phase = F.pad(vocal_phase, (0, pad_length))
                vocal_mask = F.pad(vocal_mask, (0, pad_length))

        return (
            mixture_mag.numpy(), mixture_phase.numpy(),
            instrumental_mag.numpy(), instrumental_phase.numpy(),
            vocal_mag.numpy(), vocal_phase.numpy(),
            vocal_mask.numpy()
        )

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if self.preprocess_dir:
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            data = np.load(preprocess_path)
            return (
                torch.from_numpy(data['mixture_mag']),
                torch.from_numpy(data['mixture_phase']),
                torch.from_numpy(data['instrumental_mag']),
                torch.from_numpy(data['instrumental_phase']),
                torch.from_numpy(data['vocal_mag']),
                torch.from_numpy(data['vocal_phase']),
                torch.from_numpy(data['vocal_mask'])
            )
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

def train(model, dataloader, optimizer, scheduler, loss_fn, device, epochs, checkpoint_steps, args, checkpoint_path=None):
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
        for mixture_mag, mixture_phase, instrumental_mag, instrumental_phase, vocal_mag, vocal_phase, vocal_mask in dataloader:
            # Move data to device
            mixture_mag = mixture_mag.to(device)
            mixture_phase = mixture_phase.to(device)
            instrumental_mag = instrumental_mag.to(device)
            instrumental_phase = instrumental_phase.to(device)
            vocal_mag = vocal_mag.to(device)
            vocal_phase = vocal_phase.to(device)
            vocal_mask = vocal_mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_inst_mag, pred_vocal_mag, pred_inst_phase, pred_vocal_phase = model(mixture_mag)

            # Compute loss (pass vocal_mask and segment size as additional arguments)
            loss = loss_fn(pred_inst_mag, pred_vocal_mag, pred_inst_phase, pred_vocal_phase,
                           instrumental_mag, vocal_mag, instrumental_phase, vocal_phase,
                           vocal_mask, segment_length=args.segment_length, sample_rate=44100, hop_length=args.hop_length)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update average loss and log
            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            loss_log.append(loss.item())
            step += 1
            progress_bar.update(1)

            # Update progress bar description
            desc = f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f}"
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

    # Save the final loss log
    torch.save({'loss_log': loss_log}, 'loss_log.pt')
    progress_bar.close()

def inference(model, checkpoint_path, input_wav_path, output_instrumental_path,
              chunk_size=88200, overlap=44100, device='cpu', n_fft=4096, hop_length=1024):
    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model.to(device)

    # Load the input audio
    input_audio, sr = torchaudio.load(input_wav_path)
    if input_audio.shape[0] != 2:
        raise ValueError("Input audio must have 2 channels.")
    input_audio = input_audio.to(device)

    # Initialize output tensor for instrumental
    total_length = input_audio.shape[1]
    instrumentals = torch.zeros_like(input_audio)

    # Cross-fade window for smooth blending of chunks
    cross_fade_length = overlap // 2
    window = torch.hann_window(n_fft).to(device)

    # Process the audio in chunks
    num_chunks = (total_length - overlap) // (chunk_size - overlap)
    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length - chunk_size + 1, chunk_size - overlap):
            # Extract the current chunk
            chunk = input_audio[:, i:i + chunk_size]

            # Compute the STFT of the chunk
            chunk_spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            chunk_mag = torch.abs(chunk_spec)
            chunk_phase = torch.angle(chunk_spec)

            # Add batch dimension and move to device
            chunk_mag = chunk_mag.unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():
                pred_inst_mag, _, pred_inst_phase, _ = model(chunk_mag)  # Only use instrumental outputs

            # Remove batch dimension
            pred_inst_mag = pred_inst_mag.squeeze(0)
            pred_inst_phase = pred_inst_phase.squeeze(0)

            # Use the mixture phase as a baseline and refine it with the predicted phase
            pred_inst_phase = chunk_phase + pred_inst_phase  # Add predicted phase shift

            # Combine predicted magnitude with refined phase
            pred_inst_spec = pred_inst_mag * torch.exp(1j * pred_inst_phase)

            # Reconstruct the waveform from the predicted complex spectrogram
            inst_chunk = torch.istft(pred_inst_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=chunk_size, return_complex=False)

            # Apply cross-fading for smooth blending
            if i == 0:
                # For the first chunk, don't apply cross-fading
                instrumentals[:, i:i + chunk_size] = inst_chunk
            else:
                # Apply cross-fading for subsequent chunks
                fade_in = torch.linspace(0, 1, cross_fade_length).to(device)
                fade_out = torch.linspace(1, 0, cross_fade_length).to(device)

                inst_chunk[:, :cross_fade_length] *= fade_in
                instrumentals[:, i:i + cross_fade_length] *= fade_out
                instrumentals[:, i:i + cross_fade_length] += inst_chunk[:, :cross_fade_length]

            # Fill the non-overlapping part of the chunk
            instrumentals[:, i + cross_fade_length:i + chunk_size] = inst_chunk[:, cross_fade_length:]

            pbar.update(1)

    # Clamp the output to avoid clipping
    instrumentals = torch.clamp(instrumentals, -1.0, 1.0)

    # Save the separated instrumental audio
    torchaudio.save(output_instrumental_path, instrumentals.cpu(), sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for instrumental separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='test', help='Path to training dataset')
    parser.add_argument('--preprocess_dir', type=str, default='prep', help='Path to save/load preprocessed data')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint_steps', type=int, default=2000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--segment_length', type=int, default=485100, help='Segment length for training')
    parser.add_argument('--n_fft', type=int, default=4096, help='Number of FFT bins for STFT')
    parser.add_argument('--hop_length', type=int, default=1024, help='Hop length for STFT')
    parser.add_argument('--factorization', type=str, default=None, help='Factorization type for TFNO (e.g., "tucker")')
    parser.add_argument('--rank', type=float, default=0.05, help='Rank for TFNO factorization')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralOperatorModel(in_channels=2, out_channels=4, hidden_channels=128, n_modes=(48, 48),
                                factorization=args.factorization, rank=args.rank)
    optimizer = torch.optim.Adam(model.parameters())

    if args.train:
        # Create the dataset with preprocessing if preprocess_dir is provided
        train_dataset = MUSDBDataset(root_dir=args.data_dir, preprocess_dir=args.preprocess_dir,
                                     segment_length=args.segment_length, n_fft=args.n_fft, hop_length=args.hop_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, persistent_workers=False)

        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        train(model, train_dataloader, optimizer, scheduler, loss_fn, device, args.epochs, args.checkpoint_steps, args, checkpoint_path=args.checkpoint_path)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        model = NeuralOperatorModel(in_channels=2, out_channels=4, hidden_channels=128, n_modes=(48, 48),
                                    factorization=args.factorization, rank=args.rank)
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, device=device, n_fft=args.n_fft, hop_length=args.hop_length)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
