import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import math
from x_transformers import Encoder, Decoder

class NeuralModel(nn.Module):
    def __init__(self, in_channels=2, sources=2, freq_bins=2049, max_seq_len=529200,
                embed_dim=512, depth=12, heads=12):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(freq_bins * in_channels, embed_dim)
        self.encoder = Encoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            ff_glu=True,
            rotary_pos_emb=True,
            alibi_pos_bias=True,
            alibi_num_heads=4,
            attn_pre_talking_heads=True,
            attn_post_talking_heads=True
        )
        self.decoder = Decoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            ff_glu=True,
            rotary_pos_emb=True,
            alibi_pos_bias=True,
            alibi_num_heads=4,
            attn_pre_talking_heads=True,
            attn_post_talking_heads=True
        )
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks)

    def forward(self, x):
        B, C, F, T = x.shape
        assert C == self.in_channels and F == self.freq_bins and T <= self.max_seq_len

        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_proj(x)

        x = x.view(B, T, self.out_masks, F).permute(0, 2, 3, 1)
        return x

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
    parser = argparse.ArgumentParser(description='Inference-only script for source separation')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_wav', type=str, required=True, help='Path to input WAV file')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    args = parser.parse_args()

    model = NeuralModel()
    inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=args.device)

if __name__ == '__main__':
    main()
