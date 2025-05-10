import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prodigyopt import Prodigy
from x_transformers import Encoder
import numpy as np
import random
import math
import glob
from torch.utils.checkpoint import checkpoint

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, freq_bins=2049, embed_dim=512):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.freq_bins    = freq_bins
        self.embed_dim    = embed_dim

        # project each time-slice (C * F) → embed_dim
        self.input_proj    = nn.Linear(in_channels * freq_bins, embed_dim)

        # 12-layer Transformer
        self.encoder       = Encoder(
            dim=embed_dim,
            depth=12,
            heads=8,
            ff_glu=True
        )

        # for each embed vector predict a mask and a residual,
        # each of shape [in_channels * out_channels * freq_bins]
        self.mask_proj     = nn.Linear(embed_dim,
                                       in_channels * out_channels * freq_bins)
        self.resid_proj    = nn.Linear(embed_dim,
                                       in_channels * out_channels * freq_bins)

    def forward(self, x):
        # x: [B, in_channels, freq_bins, time]
        B, C, F, T = x.shape
        assert C == self.in_channels
        assert F == self.freq_bins

        # keep the original mixture for residual path
        mixture = x  # [B, C, F, T]

        # flatten channel+freq → features per time-step
        # → [B, T, C*F]
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)

        # embed → [B, T, embed_dim]
        x = self.input_proj(x)
        # transformer → [B, T, embed_dim]
        x = self.encoder(x)

        # predict mask and residual → [B, T, in*out*F]
        mask_out  = self.mask_proj(x)
        resid_out = self.resid_proj(x)

        # reshape → [B, T, out, in, F]
        mask  = mask_out.view(B, T, self.out_channels, C, F)
        resid = resid_out.view(B, T, self.out_channels, C, F)

        # permute → [B, out, in, F, T]
        mask  = mask.permute(0, 2, 3, 4, 1)
        resid = resid.permute(0, 2, 3, 4, 1)

        # expand mixture to [B, 1, in, F, T] → broadcast to [B, out, in, F, T]
        mixture = mixture.unsqueeze(1)

        # apply mask + add residual
        out = mask * mixture + resid  # [B, out, in, F, T]
        return out

def aweight_coefficients(freqs):
    freqs = np.array(freqs)
    # avoid zero-frequency singularity
    freqs = np.maximum(freqs, 1e-6)
    numer = (12194**2) * (freqs**4)
    denom = (freqs**2 + 20.6**2) * np.sqrt(
        (freqs**2 + 107.7**2) * (freqs**2 + 737.9**2)
    ) * (freqs**2 + 12194**2)
    return 20 * np.log10(numer / (denom + 1e-10)) + 2.0

def loss_fn(pred_vocals, pred_instrumental, target_vocal_mag, target_instrumental_mag,
            scales=None, weights=None):
    device = target_vocal_mag.device
    dtype = target_vocal_mag.dtype

    # STFT params
    n_fft = 4096
    sample_rate = 44100

    # frequency bins
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    a_weights = aweight_coefficients(freqs)           # [2049]
    a_weights = torch.tensor(a_weights, device=device, dtype=dtype)
    a_weights = a_weights.view(1, 1, -1, 1)            # [1,1,2049,1]
    a_weights_linear = torch.pow(10.0, a_weights / 20.0)

    mse = torch.nn.MSELoss()
    total_loss = 0.0

    if scales  is None: scales  = [1, 2, 4, 8]
    if weights is None: weights = [1.0] * len(scales)

    for scale, w in zip(scales, weights):
        if scale == 1:
            pv, pi, tv, ti = (
                pred_vocals,
                pred_instrumental,
                target_vocal_mag,
                target_instrumental_mag
            )
            aw = a_weights_linear
        else:
            size = (
                pred_vocals.shape[2] // scale,
                pred_vocals.shape[3] // scale
            )
            pv = F.interpolate(pred_vocals,        size=size, mode='area')
            pi = F.interpolate(pred_instrumental,  size=size, mode='area')
            tv = F.interpolate(target_vocal_mag,        size=size, mode='area')
            ti = F.interpolate(target_instrumental_mag, size=size, mode='area')

            # Downsample A-weights
            aw = F.interpolate(
                a_weights_linear,
                size=(pv.shape[2], 1),
                mode='area'
            )

        # Apply perceptual A-weighting
        pv_weighted = pv * aw
        pi_weighted = pi * aw
        tv_weighted = tv * aw
        ti_weighted = ti * aw

        loss_v = mse(pv_weighted, tv_weighted)
        loss_i = mse(pi_weighted, ti_weighted)
        total_loss += w * (loss_v + loss_i)

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

def train(model, dataloader, optimizer, loss_fn, device, epochs, checkpoint_steps,
          args, checkpoint_path=None, window=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    checkpoint_files = []

    # Optionally resume
    if checkpoint_path:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        step = checkpoint_data['step']
        avg_loss = checkpoint_data['avg_loss']
        print(f"Resuming training from step {step} with average loss {avg_loss:.4f}")

    total_steps = epochs * len(dataloader)
    progress_bar = tqdm(total=total_steps, initial=step)
    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            mixture_mag, mixture_phase, vocal_mag, instrumental_mag = batch
            mixture_mag       = mixture_mag.to(device)
            vocal_mag         = vocal_mag.to(device)
            instrumental_mag  = instrumental_mag.to(device)

            optimizer.zero_grad()

            # model now outputs full spectrogram estimates [B, 2, F, T]
            pred_spec = model(mixture_mag)              
            pred_vocals = pred_spec[:, 0, :, :]         # [B, F, T]
            pred_instr  = pred_spec[:, 1, :, :]         # [B, F, T]

            # compute loss on those estimates
            loss = loss_fn(pred_vocals, pred_instr,
                           vocal_mag, instrumental_mag)

            if torch.isnan(loss).any():
                print("NaN loss detected, skipping batch")
                continue

            loss.backward()
            optimizer.step()

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            step += 1
            progress_bar.update(1)
            desc = (f"Epoch: {epoch+1}/{epochs}  "
                    f"Loss: {loss.item():.4f}  "
                    f"Avg Loss: {avg_loss:.4f}")
            progress_bar.set_description(desc)

            # checkpoint
            if step % checkpoint_steps == 0:
                ckpt = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                }
                fn = f"checkpoint_step_{step}.pt"
                torch.save(ckpt, fn)
                checkpoint_files.append(fn)
                if len(checkpoint_files) > 3:
                    old = checkpoint_files.pop(0)
                    if os.path.exists(old):
                        os.remove(old)

    progress_bar.close()

def inference(model, checkpoint_path, input_wav_path,
              output_instrumental_path, output_vocal_path,
              chunk_size=529200, overlap=88200, device='cpu'):

    # load checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval()
    model.to(device)

    # load & prep audio
    audio, sr = torchaudio.load(input_wav_path)
    if sr != 44100:
        raise ValueError(f"Input must be 44.1 kHz, got {sr}")
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]
    audio = audio.to(device)

    total_len = audio.shape[1]
    vocals       = torch.zeros_like(audio)
    instrumentals = torch.zeros_like(audio)

    n_fft     = 4096
    hop_length= 1024
    window    = torch.hann_window(n_fft).to(device)
    step_size = chunk_size - overlap
    cross_fade = overlap // 2

    num_chunks = math.ceil(max(0, total_len - overlap) / step_size)
    with tqdm(total=num_chunks, desc="Inference") as pbar:
        for start in range(0, total_len, step_size):
            end = min(start + chunk_size, total_len)
            chunk = audio[:, start:end]
            length = chunk.shape[1]

            # pad first chunk if too short
            if length < n_fft:
                if start == 0:
                    pad = n_fft - length
                    chunk = F.pad(chunk, (0, pad))
                    length = chunk.shape[1]
                else:
                    pbar.update(1)
                    continue

            # STFT
            spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length,
                              window=window, return_complex=True)
            mag  = spec.abs()
            phase= spec.angle()

            # model → full mags [1,2,F,T]
            with torch.no_grad():
                pred_spec = model(mag.unsqueeze(0)).squeeze(0)
            pred_voc_mag = pred_spec[0]   # [F, T]
            pred_i_mag   = pred_spec[1]   # [F, T]

            # reconstruct complex spectrograms
            v_spec = pred_voc_mag * torch.exp(1j * phase)
            i_spec = pred_i_mag   * torch.exp(1j * phase)

            # ISTFT per channel
            v_chunk = torch.zeros_like(chunk)
            i_chunk = torch.zeros_like(chunk)
            for ch in range(2):
                v_chunk[ch] = torch.istft(v_spec[ch].unsqueeze(0),
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          window=window,
                                          length=length).squeeze(0)
                i_chunk[ch] = torch.istft(i_spec[ch].unsqueeze(0),
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          window=window,
                                          length=length).squeeze(0)

            # overlap–add with cross-fade
            if start == 0:
                L = min(length, total_len)
                vocals[:, :L]       = v_chunk[:, :L]
                instrumentals[:, :L]= i_chunk[:, :L]
            else:
                fade_in  = torch.linspace(0, 1, cross_fade, device=device)
                fade_out = torch.linspace(1, 0, cross_fade, device=device)

                overlap_start = start
                overlap_end   = min(start + cross_fade, total_len)
                ov = overlap_end - overlap_start

                # cross-fade vocals
                v_chunk[:, :ov]    *= fade_in[:ov]
                vocals[:, overlap_start:overlap_end] *= fade_out[:ov]
                vocals[:, overlap_start:overlap_end] += v_chunk[:, :ov]

                # cross-fade instrumentals
                i_chunk[:, :ov]    *= fade_in[:ov]
                instrumentals[:, overlap_start:overlap_end] *= fade_out[:ov]
                instrumentals[:, overlap_start:overlap_end] += i_chunk[:, :ov]

                # copy remainder
                rem_start = min(start + cross_fade, total_len)
                rem_end   = min(start + length, total_len)
                if rem_start < rem_end:
                    idx0 = rem_start - start
                    idx1 = rem_end   - start
                    vocals[:, rem_start:rem_end]       = v_chunk[:, idx0:idx1]
                    instrumentals[:, rem_start:rem_end] = i_chunk[:, idx0:idx1]

            pbar.update(1)

    # clamp & save
    vocals       = vocals[:, :total_len].clamp(-1, 1).cpu()
    instrumentals= instrumentals[:, :total_len].clamp(-1, 1).cpu()
    torchaudio.save(output_vocal_path, vocals, sr)
    torchaudio.save(output_instrumental_path, instrumentals, sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for instrumental separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='train', help='Path to training dataset')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--checkpoint_steps', type=int, default=2000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--segment_length', type=int, default=529200, help='Segment length for training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)
    model = NeuralModel()
    optimizer = Prodigy(model.parameters(), lr=1.0)

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
