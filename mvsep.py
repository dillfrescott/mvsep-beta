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

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.conv(x)
        att = self.sigmoid(att)
        return x * att

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure reduction_ratio does not exceed in_channels
        reduction_ratio = min(reduction_ratio, in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class NeuralOperatorModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=64, n_modes=(16, 16), factorization=None, rank=0.05):
        super(NeuralOperatorModel, self).__init__()
        
        # Attention mechanisms
        self.spatial_attention = SpatialAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        
        # Neural operator (FNO or TFNO)
        if factorization is None:
            self.operator = FNO(n_modes=n_modes, hidden_channels=hidden_channels,
                                in_channels=in_channels, out_channels=out_channels)
        else:
            self.operator = TFNO(n_modes=n_modes, hidden_channels=hidden_channels,
                                 in_channels=in_channels, out_channels=out_channels,
                                 factorization=factorization, rank=rank)

    def forward(self, x):
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Apply the neural operator
        x = self.operator(x)
        
        return x

def compute_vocal_mask(vocal_mag, instrumental_mag, eps=1e-8):
    # Compute the ideal ratio mask (IRM)
    vocal_mask = vocal_mag / (vocal_mag + instrumental_mag + eps)
    return vocal_mask

def loss_fn(pred_vocal_mask, target_vocal_mask):
    return F.l1_loss(pred_vocal_mask, target_vocal_mask)

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
        self.window = torch.hann_window(n_fft)

        # List all tracks in the dataset
        self.tracks = [os.path.join(root_dir, track) for track in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, track))]

        # Preprocess data if preprocess_dir is provided
        if self.preprocess_dir:
            self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocess the dataset and save the results to disk.
        """
        os.makedirs(self.preprocess_dir, exist_ok=True)
        for idx, track_path in enumerate(tqdm(self.tracks, desc="Preprocessing data")):
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            if not os.path.exists(preprocess_path):
                mixture_mag, instrumental_mag, vocal_mag, vocal_mask = self._process_track(track_path)
                np.savez(preprocess_path, mixture_mag=mixture_mag,
                         instrumental_mag=instrumental_mag,
                         vocal_mag=vocal_mag, vocal_mask=vocal_mask)

    def _process_track(self, track_path):
        # Load the instrumental and vocal tracks
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

        # Convert to magnitude spectrograms
        mixture_mag = torch.abs(mixture_spec)
        instrumental_mag = torch.abs(instrumental_spec)
        vocal_mag = torch.abs(vocal_spec)

        # Compute the vocal mask (Ideal Ratio Mask)
        eps = 1e-8  # Small constant to avoid division by zero
        vocal_mask = vocal_mag / (vocal_mag + instrumental_mag + eps)

        # Optionally segment the signals
        if self.segment and self.segment_length:
            if mixture_mag.shape[2] >= self.segment_length // self.hop_length:
                start = torch.randint(0, mixture_mag.shape[2] - self.segment_length // self.hop_length, (1,))
                mixture_mag = mixture_mag[:, :, start:start + self.segment_length // self.hop_length]
                instrumental_mag = instrumental_mag[:, :, start:start + self.segment_length // self.hop_length]
                vocal_mag = vocal_mag[:, :, start:start + self.segment_length // self.hop_length]
                vocal_mask = vocal_mask[:, :, start:start + self.segment_length // self.hop_length]
            else:
                # Pad the signals if they are shorter than the segment length
                pad_length = self.segment_length // self.hop_length - mixture_mag.shape[2]
                mixture_mag = F.pad(mixture_mag, (0, pad_length))
                instrumental_mag = F.pad(instrumental_mag, (0, pad_length))
                vocal_mag = F.pad(vocal_mag, (0, pad_length))
                vocal_mask = F.pad(vocal_mask, (0, pad_length))

        return mixture_mag.numpy(), instrumental_mag.numpy(), vocal_mag.numpy(), vocal_mask.numpy()

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if self.preprocess_dir:
            preprocess_path = os.path.join(self.preprocess_dir, f'track_{idx}.npz')
            data = np.load(preprocess_path)
            mixture_mag = torch.from_numpy(data['mixture_mag'])
            instrumental_mag = torch.from_numpy(data['instrumental_mag'])
            vocal_mag = torch.from_numpy(data['vocal_mag'])
            vocal_mask = torch.from_numpy(data['vocal_mask'])
            return mixture_mag, instrumental_mag, vocal_mag, vocal_mask
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

    progress_bar = tqdm(total=epochs * len(dataloader), dynamic_ncols=True)

    model.train()
    for epoch in range(epochs):
        for mixture_mag, instrumental_mag, vocal_mag, vocal_mask in dataloader:
            mixture_mag = mixture_mag.to(device)
            vocal_mask = vocal_mask.to(device)

            optimizer.zero_grad()

            # Predict the vocal mask
            pred_vocal_mask = model(mixture_mag)
            loss = loss_fn(pred_vocal_mask, vocal_mask)

            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Calculate gradient norm
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2)
            
            # Adjust learning rate based on gradient norm
            adjust_learning_rate(optimizer, grad_norm, args.learning_rate)
            
            optimizer.step()
            scheduler.step()

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            loss_log.append(loss.item())
            step += 1

            # Fetch the current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Update the progress bar description
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.8f}"
            )
            progress_bar.update(1)  # Increment the progress bar

            if step % checkpoint_steps == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'loss_log': loss_log
                }, f"checkpoint_step_{step}.pt")

    torch.save({'loss_log': loss_log}, 'loss_log.pt')
    progress_bar.close()

def inference(model, checkpoint_path, input_wav_path, output_instrumental_path, output_vocal_path,
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

    # Initialize output tensors
    total_length = input_audio.shape[1]
    instrumentals = torch.zeros_like(input_audio)
    vocals = torch.zeros_like(input_audio)

    # Process the audio in chunks
    num_chunks = (total_length - overlap) // (chunk_size - overlap)
    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length - chunk_size + 1, chunk_size - overlap):
            chunk = input_audio[:, i:i + chunk_size]
            chunk_spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(device), return_complex=True)
            chunk_mag = torch.abs(chunk_spec)

            # Predict the vocal mask
            with torch.no_grad():
                pred_vocal_mask = model(chunk_mag.unsqueeze(0)).squeeze(0)

            # Separate vocals and instrumental using the predicted mask
            pred_vocal_spec = pred_vocal_mask * chunk_spec
            pred_inst_spec = (1 - pred_vocal_mask) * chunk_spec

            # Reconstruct the waveforms
            vocal_chunk = torch.istft(pred_vocal_spec, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(device), length=chunk_size, return_complex=False)
            inst_chunk = torch.istft(pred_inst_spec, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(device), length=chunk_size, return_complex=False)

            # Handle the first chunk separately to avoid cutting off the beginning
            if i == 0:
                vocals[:, :chunk_size] = vocal_chunk
                instrumentals[:, :chunk_size] = inst_chunk
            else:
                # Apply cross-fading for smooth blending
                fade_in = torch.linspace(0, 1, overlap // 2).to(device)
                fade_out = torch.linspace(1, 0, overlap // 2).to(device)
                vocal_chunk[:, :overlap // 2] *= fade_in
                vocals[:, i:i + overlap // 2] *= fade_out
                vocals[:, i:i + overlap // 2] += vocal_chunk[:, :overlap // 2]
                inst_chunk[:, :overlap // 2] *= fade_in
                instrumentals[:, i:i + overlap // 2] *= fade_out
                instrumentals[:, i:i + overlap // 2] += inst_chunk[:, :overlap // 2]

                # Fill the non-overlapping part of the chunk
                vocals[:, i + overlap // 2:i + chunk_size] = vocal_chunk[:, overlap // 2:]
                instrumentals[:, i + overlap // 2:i + chunk_size] = inst_chunk[:, overlap // 2:]

            pbar.update(1)

    # Save the separated audio
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

    # Update out_channels to match the target vocal mask
    model = NeuralOperatorModel(in_channels=2, out_channels=2, hidden_channels=128, n_modes=(48, 48),
                                factorization=args.factorization, rank=args.rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.train:
        # Create the dataset with preprocessing if preprocess_dir is provided
        train_dataset = MUSDBDataset(root_dir=args.data_dir, preprocess_dir=args.preprocess_dir,
                                     segment_length=args.segment_length, n_fft=args.n_fft, hop_length=args.hop_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=16, pin_memory=False, persistent_workers=True)

        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        train(model, train_dataloader, optimizer, scheduler, loss_fn, device, args.epochs, args.checkpoint_steps, args, checkpoint_path=args.checkpoint_path)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        # Update out_channels to match the target vocal mask
        model = NeuralOperatorModel(in_channels=2, out_channels=2, hidden_channels=128, n_modes=(48, 48),
                                    factorization=args.factorization, rank=args.rank)
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device, n_fft=args.n_fft, hop_length=args.hop_length)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
