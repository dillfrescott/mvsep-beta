import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prodigyopt import Prodigy
import random
import math
from einops import rearrange

def apply_rope(x, theta=10000):
    b, h, n, d, device = *x.shape, x.device
    seq = torch.arange(n, device=device)
    theta_values = 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))
    freqs = torch.einsum('n,d->nd', seq, theta_values)
    freqs_cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)

    rotated_x1 = x1 * freqs_cos - x2 * freqs_sin
    rotated_x2 = x1 * freqs_sin + x2 * freqs_cos

    return torch.cat((rotated_x1, rotated_x2), dim=-1)

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

        q = apply_rope(q)
        k = apply_rope(k)

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
    def __init__(self, in_channels=2, sources=2, freq_bins=2049, max_seq_len=529200,
                 embed_dim=1024):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(freq_bins * in_channels, embed_dim)

        self.trf1 = Transformer(embed_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.trf2 = Transformer(embed_dim)

        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks)

    def forward(self, x):
        B, C, F, T = x.shape
        assert C == self.in_channels and F == self.freq_bins and T <= self.max_seq_len

        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x = self.input_proj(x)
        x = self.trf1(x)
        x = self.bottleneck(x)
        x = self.trf2(x)
        x = self.output_proj(x)

        x = x.view(B, T, self.out_masks, F).permute(0, 2, 3, 1)
        return x

def a_weighting(f):
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    f_sq = f**2
    A_num = (f4**2) * (f_sq)**2
    A_den = (f_sq + f1**2) * torch.sqrt((f_sq + f2**2)*(f_sq + f3**2)) * (f_sq + f4**2)
    A = 2.0 + 20.0 * torch.log10(A_num / (A_den + 1e-20))
    return A

def loss_fn(pred_masks,
            target_vocal,
            target_instr,
            mixture_mag,
            sr: int = 44100,
            n_fft: int = 4096,
            eps: float = 1e-7):

    # 1) Build A‑weighting per‑frequency linear gains
    device = mixture_mag.device
    dtype  = mixture_mag.dtype
    freqs = torch.linspace(0, sr/2, n_fft//2 + 1, device=device, dtype=dtype)
    A_db  = a_weighting(freqs)
    A_lin = 10 ** (A_db / 20.0)
    weight_f = A_lin.view(1, 1, -1, 1)

    # 2) Split predicted masks
    vL, vR, iL, iR = pred_masks.chunk(4, dim=1)  # each [B,1,F,T]

    # 3) Reconstruct magnitude estimates
    pred_v = torch.cat([vL * mixture_mag[:,0:1],
                        vR * mixture_mag[:,1:2]], dim=1)  # [B,2,F,T]
    pred_i = torch.cat([iL * mixture_mag[:,0:1],
                        iR * mixture_mag[:,1:2]], dim=1)

    # 4a) Compute A‑weighted MSE
    sq_err_v = (pred_v - target_vocal)**2  # [B,2,F,T]
    sq_err_i = (pred_i - target_instr)**2
    weighted_v = weight_f * sq_err_v
    weighted_i = weight_f * sq_err_i
    weighted_loss = weighted_v.mean() + weighted_i.mean()

    # 4b) Compute unweighted MSE
    unweighted_loss = sq_err_v.mean() + sq_err_i.mean()

    # 5) Combine unweighted + weighted
    final_loss = 0.5 * weighted_loss + 0.5 * unweighted_loss
    return final_loss

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

        # Lists for vocals and "other" stems
        self.vocal_paths = []
        self.other_paths = []

        self.track_dirs = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]

        print("Scanning track folders...")
        for td in tqdm(self.track_dirs, desc="Scanning tracks"):
            vocal_path = os.path.join(td, 'vocals.wav')
            other_path = os.path.join(td, 'other.wav')

            if os.path.exists(vocal_path):
                self.vocal_paths.append(vocal_path)
            if os.path.exists(other_path):
                self.other_paths.append(other_path)

        if not self.vocal_paths or not self.other_paths:
            raise ValueError("Dataset must contain both vocal and 'other' stems.")

        self.size = 50000  # Arbitrary large number for training

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
        if min_length == 0:
            raise ValueError("Encountered an audio file with zero length.")

        if self.segment and self.segment_length < min_length:
            start = random.randint(0, min_length - self.segment_length)
            end = start + self.segment_length
        else:
            start = 0
            end = min_length

        vocal_seg = vocal_audio[:, start:end]
        instr_seg = instr_audio[:, start:end]

        vocal_spec = torch.stft(vocal_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window, return_complex=True)
        instr_spec = torch.stft(instr_seg, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window, return_complex=True)

        vocal_mag = torch.abs(vocal_spec)
        instr_mag = torch.abs(instr_spec)

        mixture_spec = vocal_spec + instr_spec
        mixture_mag = torch.abs(mixture_spec)
        mixture_phase = torch.angle(mixture_spec)

        target_time_bins = 1 + (self.segment_length - self.n_fft) // self.hop_length

        def pad_or_trim(x, target_len):
            current_len = x.shape[-1]
            if current_len < target_len:
                pad_amt = target_len - current_len
                return F.pad(x, (0, pad_amt))
            elif current_len > target_len:
                return x[..., :target_len]
            return x

        mixture_mag = pad_or_trim(mixture_mag, target_time_bins)
        vocal_mag = pad_or_trim(vocal_mag, target_time_bins)
        instr_mag = pad_or_trim(instr_mag, target_time_bins)
        mixture_phase = pad_or_trim(mixture_phase, target_time_bins)

        return mixture_mag, mixture_phase, vocal_mag, instr_mag

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
            loss = loss_fn(pred_vocal_mask, vocal_mag, instrumental_mag, mixture_mag)
            
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
    # Load checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval().to(device)

    # Load input audio
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

    # STFT params
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
                              window=window, return_complex=True)  # [2, F, T]
            mag = torch.abs(spec)
            phase = torch.angle(spec)

            with torch.no_grad():
                pred_masks = model(mag.unsqueeze(0))  # [1, 4, F, T]
            pred_masks = pred_masks.squeeze(0)       # [4, F, T]

            vL, vR, iL, iR = (m.squeeze(0) for m in pred_masks.chunk(4, dim=0))

            # Apply masks
            v_spec = torch.stack([vL * mag[0], vR * mag[1]], dim=0) * torch.exp(1j * phase)
            i_spec = torch.stack([iL * mag[0], iR * mag[1]], dim=0) * torch.exp(1j * phase)

            # Inverse STFT
            vocal_chunk = torch.istft(v_spec, n_fft=n_fft, hop_length=hop_length,
                                      window=window, length=L)
            inst_chunk = torch.istft(i_spec, n_fft=n_fft, hop_length=hop_length,
                                     window=window, length=L)

            # Overlap-add
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
    model = NeuralModel()
    optimizer = Prodigy(model.parameters(), lr=1.0)

    if args.train:
        train_dataset = MUSDBDataset(root_dir=args.data_dir,
                                     segment_length=args.segment_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=24, pin_memory=True, persistent_workers=True)
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
