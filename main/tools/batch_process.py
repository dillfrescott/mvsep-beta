import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast
import mvsep
from mvsep import NeuralModel, clean_state_dict, inference as mvsep_inference

warnings.filterwarnings("ignore")

def inference(model, checkpoint_data, input_dir, output_dir, chunk_size=529200, overlap=44100, device='cpu'):
    stems = checkpoint_data.get('stems')
    if stems is None:
        sources = model.sources if hasattr(model, 'sources') else 2
        if sources == 2:
            stems = ['vocals', 'other']
        else:
            stems = [f'stem_{i}' for i in range(sources)]
            
    mvsep.STEMS = stems
    num_stems = len(stems)
    if 'ema_state_dict' in checkpoint_data:
        print("Loading EMA weights for inference.")
        model.load_state_dict(clean_state_dict(checkpoint_data['ema_state_dict']), strict=False)
    else:
        print("Loading regular model weights for inference.")
        model.load_state_dict(clean_state_dict(checkpoint_data['model_state_dict']), strict=False)
    model.eval().to(device)

    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.flac'))]
    input_files.sort()

    for filename in input_files:
        input_path = os.path.join(input_dir, filename)
        wav_name = os.path.splitext(filename)[0]
        song_output_dir = os.path.join(output_dir, wav_name)
        os.makedirs(song_output_dir, exist_ok=True)

        input_audio, sr = torchaudio.load(input_path)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
            input_audio = resampler(input_audio)
            sr = 44100
        if input_audio.shape[0] == 1:
            input_audio = input_audio.repeat(2, 1)
        elif input_audio.shape[0] != 2:
            continue
        input_audio = input_audio.to(device)

        with torch.no_grad():
            pred_stems = mvsep_inference(
                model=model,
                checkpoint_path=None,
                input_data=input_audio,
                chunk_size=chunk_size,
                overlap=overlap,
                device=device,
                return_tensors=True
            )

        residual = input_audio - sum(pred_stems)
        for j in range(num_stems):
            res = (pred_stems[j] + (1.0 / num_stems) * residual).clamp(-1.0, 1.0)
            torchaudio.save(os.path.join(song_output_dir, f'{stems[j]}.flac'), res.cpu(), sr, format='flac')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_data = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint_data.get('model_state_dict', checkpoint_data.get('ema_state_dict', {}))
    sources = 3
    for k in state_dict.keys():
        if k.endswith('proj_to_pixel_shuffle.weight'):
            out_channels = state_dict[k].shape[0]
            sources = out_channels // 48
            break
            
    stems = checkpoint_data.get('stems')
    if stems is None:
        if sources == 2:
            stems = ['vocals', 'other']
        else:
            stems = [f'stem_{i}' for i in range(sources)]
            
    mvsep.STEMS = stems
    model = NeuralModel(sources=sources)

    if args.infer:
        inference(model, checkpoint_data, args.input_dir, args.output_dir, device=device)

if __name__ == '__main__':
    main()
