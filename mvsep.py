import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prodigyopt import Prodigy
import random
import math
from einops import rearrange
from nnAudio.features import VQT

class HarmonicBranch(nn.Module):
    def __init__(self, sample_rate=44100, n_bins=192, embed_dim=1024, hop_length=1024):
        super().__init__()
        self.vqt_transform = VQT(
            sr=sample_rate,
            hop_length=hop_length,
            fmin=librosa.note_to_hz('C1'),
            n_bins=n_bins,
            bins_per_octave=24,
            gamma=20,
            output_format='Complex'
        )
        self.vqt_proj = nn.Linear(n_bins, embed_dim)

    def forward(self, x_audio):
        x_mono = torch.mean(x_audio, dim=1)
        vqt_spec = self.vqt_transform(x_mono)
        vqt_mag = torch.norm(vqt_spec, p=2, dim=-1)
        vqt_mag_permuted = vqt_mag.permute(0, 2, 1)
        return self.vqt_proj(vqt_mag_permuted)

def apply_2d_rope(x, chunk_size=256, theta=10000):
    b, h, n, d, device = *x.shape, x.device

    num_chunks = n // chunk_size
    local_pos = n % chunk_size
    if local_pos != 0:
        padding = chunk_size - local_pos
        x = F.pad(x, (0, 0, 0, padding))
        n = n + padding
        num_chunks = n // chunk_size
    
    x = rearrange(x, 'b h (c l) d -> b h c l d', c=num_chunks)

    seq_local = torch.arange(chunk_size, device=device)
    theta_local = 1.0 / (theta ** (torch.arange(0, d, 4, device=device).float() / d))
    freqs_local = torch.einsum('l,d->ld', seq_local, theta_local)
    freqs_cos_local = freqs_local.cos().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    freqs_sin_local = freqs_local.sin().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    x1, x2, x3, x4 = x.chunk(4, dim=-1)
    
    rotated_x1_local = x1 * freqs_cos_local - x2 * freqs_sin_local
    rotated_x2_local = x1 * freqs_sin_local + x2 * freqs_cos_local

    seq_chunk = torch.arange(num_chunks, device=device)
    theta_chunk = 1.0 / (theta ** (torch.arange(0, d, 4, device=device).float() / d))
    freqs_chunk = torch.einsum('c,d->cd', seq_chunk, theta_chunk)
    freqs_cos_chunk = freqs_chunk.cos().unsqueeze(0).unsqueeze(0).unsqueeze(-2)
    freqs_sin_chunk = freqs_chunk.sin().unsqueeze(0).unsqueeze(0).unsqueeze(-2)

    rotated_x3_chunk = x3 * freqs_cos_chunk - x4 * freqs_sin_chunk
    rotated_x4_chunk = x3 * freqs_sin_chunk + x4 * freqs_cos_chunk

    x_rotated = torch.cat((rotated_x1_local, rotated_x2_local, rotated_x3_chunk, rotated_x4_chunk), dim=-1)
    
    x_rotated = rearrange(x_rotated, 'b h c l d -> b h (c l) d')

    if 'padding' in locals() and padding > 0:
        x_rotated = x_rotated[:, :, :-padding, :]

    return x_rotated

class TalkingHeadAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.pre_softmax_proj = nn.Linear(heads, heads, bias=False)
        self.post_softmax_proj = nn.Linear(heads, heads, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = apply_2d_rope(q)
        k = apply_2d_rope(k)

        attn_logits = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        attn_logits = rearrange(attn_logits, 'b h i j -> b i j h')
        attn_logits = self.pre_softmax_proj(attn_logits)
        attn_logits = rearrange(attn_logits, 'b i j h -> b h i j')

        if mask is not None:
            mask_value = -torch.finfo(attn_logits.dtype).max
            attn_logits = attn_logits.masked_fill(mask, mask_value)

        attn = attn_logits.softmax(dim=-1)

        attn = rearrange(attn, 'b h i j -> b i j h')
        attn = self.post_softmax_proj(attn)
        attn = rearrange(attn, 'b i j h -> b h i j')

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = TalkingHeadAttention(dim, heads)
        self.ff = FeedForward(dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, mask=mask)))
        x = self.norm2(x + self.ff(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim, heads=8, num_layers=6, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dim, heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049,
                 embed_dim=1024, sample_rate=44100, hop_length=1024):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels

        self.input_proj_stft = nn.Linear(freq_bins * in_channels, embed_dim)

        self.harmonic_branch = HarmonicBranch(
            sample_rate=sample_rate,
            embed_dim=embed_dim,
            hop_length=hop_length
        )

        self.trf1 = Transformer(embed_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.trf2 = Transformer(embed_dim)

        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks * 2)

    def forward(self, x_stft_mag, x_audio):
        B, C, F, T = x_stft_mag.shape

        x_stft_mag = x_stft_mag.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        projected_stft = self.input_proj_stft(x_stft_mag)

        projected_vqt = self.harmonic_branch(x_audio)

        if projected_stft.shape[1] != projected_vqt.shape[1]:
            min_T = min(projected_stft.shape[1], projected_vqt.shape[1])
            projected_stft = projected_stft[:, :min_T, :]
            projected_vqt = projected_vqt[:, :min_T, :]
        
        x = projected_stft + projected_vqt

        x = self.trf1(x)
        x = self.bottleneck(x)
        x = self.trf2(x)
        x = self.output_proj(x)
        
        current_T = x.shape[1]
        x = x.view(B, current_T, self.out_masks * 2, F).permute(0, 2, 3, 1)

        if current_T < T:
            pad_amount = T - current_T
            x = F.pad(x, (0, pad_amount))

        return x

class MultiResolutionComplexSTFTLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super(MultiResolutionComplexSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        for i, win_len in enumerate(win_lengths):
            self.register_buffer(f'window_{i}', torch.hann_window(win_len), persistent=False)

    def forward(self, y_pred, y_true):
        complex_loss_total = 0.0

        if y_pred.dim() == 3:
            y_pred = y_pred.reshape(-1, y_pred.size(2))
            y_true = y_true.reshape(-1, y_true.size(2))

        for i, (n_fft, hop_length, win_length) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f'window_{i}')

            stft_pred = torch.stft(y_pred, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            stft_true = torch.stft(y_true, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

            real_loss = F.mse_loss(stft_pred.real, stft_true.real)
            imag_loss = F.mse_loss(stft_pred.imag, stft_true.imag)
            
            complex_loss_total += (real_loss + imag_loss)

        return complex_loss_total

def loss_fn(pred_output,
            mixture_spec,
            target_vocal_audio,
            target_instr_audio,
            stft_params_for_istft):
    
    multi_res_complex_loss_calculator = MultiResolutionComplexSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192]
    ).to(pred_output.device)

    B, _, F, T = pred_output.shape
    pred_masks = pred_output.view(B, 2, 4, F, T)
    pred_masks_real = pred_masks[:, 0]
    pred_masks_imag = pred_masks[:, 1]

    vL_cmask = pred_masks_real[:, 0] + 1j * pred_masks_imag[:, 0]
    vR_cmask = pred_masks_real[:, 1] + 1j * pred_masks_imag[:, 1]
    iL_cmask = pred_masks_real[:, 2] + 1j * pred_masks_imag[:, 2]
    iR_cmask = pred_masks_real[:, 3] + 1j * pred_masks_imag[:, 3]

    vL_cmask, vR_cmask = vL_cmask.unsqueeze(1), vR_cmask.unsqueeze(1)
    iL_cmask, iR_cmask = iL_cmask.unsqueeze(1), iR_cmask.unsqueeze(1)

    v_spec_pred = torch.cat([vL_cmask * mixture_spec[:, 0:1],
                             vR_cmask * mixture_spec[:, 1:2]], dim=1)
    i_spec_pred = torch.cat([iL_cmask * mixture_spec[:, 0:1],
                             iR_cmask * mixture_spec[:, 1:2]], dim=1)

    n_fft = stft_params_for_istft['n_fft']
    hop_length = stft_params_for_istft['hop_length']
    window = stft_params_for_istft['window'].to(pred_output.device)
    recon_len = target_vocal_audio.shape[-1]

    B, C, freq, T_spec = v_spec_pred.shape
    v_spec_pred_reshaped = v_spec_pred.reshape(B * C, freq, T_spec)
    i_spec_pred_reshaped = i_spec_pred.reshape(B * C, freq, T_spec)

    pred_vocal_audio = torch.istft(
        v_spec_pred_reshaped, n_fft=n_fft, hop_length=hop_length,
        window=window, center=True, length=recon_len
    ).reshape(B, C, -1)
    
    pred_instr_audio = torch.istft(
        i_spec_pred_reshaped, n_fft=n_fft, hop_length=hop_length,
        window=window, center=True, length=recon_len
    ).reshape(B, C, -1)

    vocal_loss = multi_res_complex_loss_calculator(pred_vocal_audio, target_vocal_audio)
    instr_loss = multi_res_complex_loss_calculator(pred_instr_audio, target_instr_audio)
    
    total_loss = vocal_loss + instr_loss

    return total_loss

class Dataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_length=88200, segment=True):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment = segment

        self.n_fft = 4096
        self.hop_length = 1024
        self.window = torch.hann_window(self.n_fft)

        self.vocal_paths = []
        self.other_paths = []

        track_dirs = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]

        print("Scanning track folders...")
        for td in tqdm(track_dirs, desc="Scanning tracks"):
            vocal_path = os.path.join(td, 'vocals.wav')
            other_path = os.path.join(td, 'other.wav')

            if os.path.exists(vocal_path):
                self.vocal_paths.append(vocal_path)
            if os.path.exists(other_path):
                self.other_paths.append(other_path)

        if not self.vocal_paths or not self.other_paths:
            raise ValueError("Dataset must contain both vocal and 'other' stems.")

        self.size = 50000

    def _preprocess_audio(self, audio, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        return audio

    def _load_audio(self, filepath):
        audio, sr = torchaudio.load(filepath)
        return self._preprocess_audio(audio, sr)

    def _load_vocal(self, path):
        return self._load_audio(path)

    def _load_instrumental(self, path):
        return self._load_audio(path)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        vocal_path = random.choice(self.vocal_paths)
        other_path = random.choice(self.other_paths)

        vocal_audio = self._load_vocal(vocal_path)
        instr_audio = self._load_instrumental(other_path)

        min_length = min(vocal_audio.shape[1], instr_audio.shape[1])
        if min_length < self.segment_length:
            raise ValueError(f"Encountered an audio file shorter than the segment length. Min length: {min_length}")

        if self.segment:
            start = random.randint(0, min_length - self.segment_length)
            end = start + self.segment_length
            vocal_seg = vocal_audio[:, start:end]
            instr_seg = instr_audio[:, start:end]
        else:
            vocal_seg = vocal_audio
            instr_seg = instr_audio

        mixture_seg = vocal_seg + instr_seg

        mixture_spec = torch.stft(mixture_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=self.window, return_complex=True, center=True)

        return mixture_spec, vocal_seg, instr_seg, mixture_seg

def train(model, dataloader, optimizer, loss_fn, device, epochs, checkpoint_steps, args, checkpoint_path=None, window=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    checkpoint_files = []

    stft_params_for_istft = {
        'n_fft': 4096,
        'hop_length': 1024,
        'window': window.to(device)
    }

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
            mixture_spec, vocal_audio, instr_audio, mixture_audio = batch
            
            mixture_mag = torch.abs(mixture_spec).to(device)
            mixture_spec = mixture_spec.to(device)
            vocal_audio = vocal_audio.to(device)
            instr_audio = instr_audio.to(device)
            mixture_audio = mixture_audio.to(device)

            optimizer.zero_grad()
            pred_masks = model(mixture_mag, mixture_audio)

            loss = loss_fn(pred_masks,
                           mixture_spec,
                           vocal_audio,
                           instr_audio,
                           stft_params_for_istft)
            
            if torch.isnan(loss).any():
                print("NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            step += 1
            progress_bar.update(1)
            current_lr = optimizer.param_groups[0]['lr']
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
              chunk_size=529200, overlap=88200, device='cpu'):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval().to(device)

    input_audio, sr = torchaudio.load(input_wav_path)
    if sr != 44100:
        raise ValueError(f"Input audio must be 44100Hz, but got {sr}Hz.")
    if input_audio.shape[0] == 1:
        input_audio = input_audio.repeat(2, 1)
    elif input_audio.shape[0] != 2:
        raise ValueError("Input audio must be mono or stereo.")
    input_audio = input_audio.to(device)

    total_length = input_audio.shape[1]
    vocals = torch.zeros_like(input_audio)
    instrumentals = torch.zeros_like(input_audio)

    n_fft = 4096
    hop_length = 1024
    window = torch.hann_window(n_fft).to(device)
    min_chunk = n_fft

    step_size = max(1, chunk_size - overlap)
    cross_fade = overlap // 2
    num_chunks = math.ceil(max(0, total_length - overlap) / step_size)

    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length, step_size):
            end = min(i + chunk_size, total_length)
            chunk = input_audio[:, i:end]
            L = chunk.shape[1]

            if L < min_chunk:
                if i == 0:
                    pad_amt = min_chunk - L
                    chunk = F.pad(chunk, (0, pad_amt))
                    L = chunk.shape[1]
                else:
                    pbar.update(1)
                    continue

            spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length,
                              window=window, return_complex=True, center=True)
            mag = torch.abs(spec)

            with torch.no_grad():
                pred_output = model(mag.unsqueeze(0), chunk.unsqueeze(0))
            pred_output = pred_output.squeeze(0)

            _, F_spec, T_spec = pred_output.shape
            pred_masks = pred_output.view(2, 4, F_spec, T_spec)
            pred_masks_real = pred_masks[0]
            pred_masks_imag = pred_masks[1]

            vL_cmask = pred_masks_real[0] + 1j * pred_masks_imag[0]
            vR_cmask = pred_masks_real[1] + 1j * pred_masks_imag[1]
            iL_cmask = pred_masks_real[2] + 1j * pred_masks_imag[2]
            iR_cmask = pred_masks_real[3] + 1j * pred_masks_imag[3]

            instrumental_spec = torch.stack([iL_cmask * spec[0], iR_cmask * spec[1]], dim=0)
            vocal_spec = torch.stack([vL_cmask * spec[0], vR_cmask * spec[1]], dim=0)

            vocal_chunk = torch.istft(vocal_spec, n_fft=n_fft, hop_length=hop_length,
                                      window=window, length=L, center=True)
            inst_chunk = torch.istft(instrumental_spec, n_fft=n_fft, hop_length=hop_length,
                                     window=window, length=L, center=True)

            if i == 0:
                vocals[:, :L] = vocal_chunk
                instrumentals[:, :L] = inst_chunk
            else:
                fade_in = torch.linspace(0, 1, cross_fade, device=device)
                fade_out = 1 - fade_in
                ov_end = min(i + cross_fade, total_length)
                actual = ov_end - i

                vocals[:, i:ov_end] = vocals[:, i:ov_end] * fade_out[:actual] + vocal_chunk[:, :actual] * fade_in[:actual]
                instrumentals[:, i:ov_end] = instrumentals[:, i:ov_end] * fade_out[:actual] + inst_chunk[:, :actual] * fade_in[:actual]

                tail_start = i + cross_fade
                tail_end = min(i + L, total_length)
                if tail_start < tail_end:
                    vocals[:, tail_start:tail_end] = vocal_chunk[:, tail_start - i:tail_end - i]
                    instrumentals[:, tail_start:tail_end] = inst_chunk[:, tail_start - i:tail_end - i]

            pbar.update(1)

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
    parser.add_argument('--segment_length', type=int, default=529200, help='Segment length for training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(4096).to(device)

    model = NeuralModel(sample_rate=44100, hop_length=1024)
    
    optimizer = Prodigy(model.parameters(), lr=1.0)

    if args.train:
        train_dataset = Dataset(root_dir=args.data_dir,
                                      segment_length=args.segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=24, pin_memory=False, persistent_workers=True)
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
