import os
import random
import cupy as cp
import numpy as np
import string
import librosa
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Paths and parameters
dataset_path = "train"
output_path = "augmented_train"
num_combinations = 50
batch_size = 10
desired_length_sec = 4 * 60  # 4 minutes
chunk_length_sec = 60  # 1 minute
sr = 44100  # Sample rate

os.makedirs(output_path, exist_ok=True)

def get_all_files(dataset_path):
    vocals_files = []
    other_files = []
    for song_dir in os.listdir(dataset_path):
        song_path = os.path.join(dataset_path, song_dir)
        if os.path.isdir(song_path):
            vocals_path = os.path.join(song_path, "vocals.wav")
            other_path = os.path.join(song_path, "other.wav")
            if os.path.exists(vocals_path) and os.path.exists(other_path):
                vocals_files.append(os.path.abspath(vocals_path))
                other_files.append(os.path.abspath(other_path))
    return vocals_files, other_files

def random_song_name(length=10):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def apply_loudness(audio_chunk, min_gain=0.2, max_gain=2.0):
    gain_multiplier = random.uniform(min_gain, max_gain)
    return audio_chunk * gain_multiplier

def apply_mixup(audio_chunk, additional_stems, min_gain=0.2, max_gain=2.0):
    if random.random() < 0.2:
        mix_stem = random.choice(additional_stems)
        # Ensure mix_stem is same length
        if mix_stem.shape[1] < audio_chunk.shape[1]:
            mix_stem = cp.pad(mix_stem, ((0, 0), (0, audio_chunk.shape[1] - mix_stem.shape[1])))
        elif mix_stem.shape[1] > audio_chunk.shape[1]:
            mix_stem = mix_stem[:, :audio_chunk.shape[1]]
        mix_gain = random.uniform(min_gain, max_gain)
        audio_chunk = audio_chunk + (mix_stem * mix_gain)
    return audio_chunk

def apply_random_inversion(audio_chunk):
    if random.random() < 0.1:
        return -audio_chunk  # Phase flip
    elif random.random() < 0.05:
        return audio_chunk[:, ::-1]  # Reverse
    return audio_chunk

def apply_random_pitch_shift(audio_chunk, sr, min_semitones=-6, max_semitones=6, prob=0.1):
    if random.random() < prob:
        semitones = random.uniform(min_semitones, max_semitones)
        # Convert CuPy -> NumPy for Librosa
        audio_np = cp.asnumpy(audio_chunk)
        # Process channels separately since librosa expects 1D
        processed = []
        for ch in range(audio_np.shape[0]):
            channel_data = audio_np[ch]
            channel_data = librosa.effects.pitch_shift(channel_data, sr=sr, n_steps=semitones)
            stretch_factor = 2 ** (semitones / 12.0)
            channel_data = librosa.effects.time_stretch(channel_data, rate=stretch_factor)
            processed.append(channel_data)
        processed_audio = np.stack(processed)
        # Convert back to CuPy
        return cp.asarray(processed_audio)
    return audio_chunk

def apply_lowpass_sweep_stft(audio, sr, start_cutoff=15000, end_cutoff=5, n_fft=2048, hop_length=512, margin=1000):
    channels, total_samples = audio.shape
    # Use NumPy for frequency calculation
    freqs = np.asarray(librosa.fft_frequencies(sr=sr, n_fft=n_fft))
    
    # Convert to NumPy for processing
    audio_np = cp.asnumpy(audio)
    processed_audio = np.zeros_like(audio_np)
    
    for ch in range(channels):
        stft = librosa.stft(audio_np[ch], n_fft=n_fft, hop_length=hop_length)
        num_frames = stft.shape[1]
        mask = np.zeros_like(stft, dtype=np.float32)
        
        for i in range(num_frames):
            t = i / num_frames
            cutoff = start_cutoff + t * (end_cutoff - start_cutoff)
            frame_mask = np.ones_like(freqs)
            frame_mask[freqs > cutoff + margin] = 0.0
            between = (freqs >= cutoff) & (freqs <= cutoff + margin)
            frame_mask[between] = (cutoff + margin - freqs[between]) / margin
            mask[:, i] = frame_mask
        
        processed_stft = stft * mask
        processed_audio_ch = librosa.istft(processed_stft, hop_length=hop_length, length=total_samples)
        processed_audio[ch, :len(processed_audio_ch)] = processed_audio_ch
    
    # Convert back to CuPy
    return cp.asarray(processed_audio)

def process_chunk(vocals_chunk, other_chunk, additional_vocals_segments, additional_others_segments, sr):
    vocals_chunk = apply_loudness(vocals_chunk)
    vocals_chunk = apply_mixup(vocals_chunk, additional_vocals_segments)
    vocals_chunk = apply_random_inversion(vocals_chunk)
    vocals_chunk = apply_random_pitch_shift(vocals_chunk, sr)
    vocals_chunk = apply_lowpass_sweep_stft(vocals_chunk, sr)
    
    other_chunk = apply_loudness(other_chunk)
    other_chunk = apply_mixup(other_chunk, additional_others_segments)
    other_chunk = apply_random_inversion(other_chunk)
    other_chunk = apply_random_pitch_shift(other_chunk, sr)
    other_chunk = apply_lowpass_sweep_stft(other_chunk, sr)
    
    return vocals_chunk, other_chunk

def combine_chunks(processed_vocals_chunks, processed_other_chunks, desired_length_samples):
    final_vocals = cp.concatenate(processed_vocals_chunks, axis=1)
    final_other = cp.concatenate(processed_other_chunks, axis=1)
    
    # Trim or pad to desired length
    if final_vocals.shape[1] > desired_length_samples:
        final_vocals = final_vocals[:, :desired_length_samples]
    else:
        final_vocals = cp.pad(final_vocals, ((0, 0), (0, desired_length_samples - final_vocals.shape[1])))
    
    if final_other.shape[1] > desired_length_samples:
        final_other = final_other[:, :desired_length_samples]
    else:
        final_other = cp.pad(final_other, ((0, 0), (0, desired_length_samples - final_other.shape[1])))
    
    return final_vocals, final_other

def create_combination(combination_data):
    vocals_path, other_path, output_dir, additional_vocals, additional_others = combination_data
    
    try:
        chunk_length_samples = int(chunk_length_sec * sr)
        desired_length_samples = int(desired_length_sec * sr)
        
        # Load and convert all audio to CuPy arrays
        def load_and_prepare(paths):
            segments = []
            for path in paths:
                # Load with librosa as NumPy
                audio, _ = librosa.load(path, sr=sr, mono=False, dtype=np.float32)
                if audio.ndim == 1:
                    audio = np.stack([audio, audio])  # Convert mono to stereo
                # Convert to CuPy
                audio = cp.asarray(audio)
                # Pad or trim to chunk length
                if audio.shape[1] < chunk_length_samples:
                    audio = cp.pad(audio, ((0, 0), (0, chunk_length_samples - audio.shape[1])))
                else:
                    audio = audio[:, :chunk_length_samples]
                segments.append(audio)
            return segments
        
        additional_vocals_segments = load_and_prepare(additional_vocals)
        additional_others_segments = load_and_prepare(additional_others)
        
        # Load main vocals and other
        vocals, _ = librosa.load(vocals_path, sr=sr, mono=False, dtype=np.float32)
        if vocals.ndim == 1:
            vocals = np.stack([vocals, vocals])
        # Convert to CuPy
        vocals = cp.asarray(vocals)
        if vocals.shape[1] < chunk_length_samples:
            vocals = cp.pad(vocals, ((0, 0), (0, chunk_length_samples - vocals.shape[1])))
        
        other, _ = librosa.load(other_path, sr=sr, mono=False, dtype=np.float32)
        if other.ndim == 1:
            other = np.stack([other, other])
        # Convert to CuPy
        other = cp.asarray(other)
        if other.shape[1] < chunk_length_samples:
            other = cp.pad(other, ((0, 0), (0, chunk_length_samples - other.shape[1])))
        
        # Process chunks sequentially
        num_chunks_needed = int(desired_length_samples // chunk_length_samples)
        processed_vocals_chunks = []
        processed_other_chunks = []
        
        for _ in range(num_chunks_needed):
            if random.random() < 0.5:
                vocals_chunk = vocals[:, :chunk_length_samples]
                other_chunk = other[:, :chunk_length_samples]
            else:
                vocals_chunk = random.choice(additional_vocals_segments)
                other_chunk = random.choice(additional_others_segments)
            
            # Ensure chunk length
            if vocals_chunk.shape[1] < chunk_length_samples:
                vocals_chunk = cp.pad(vocals_chunk, ((0, 0), (0, chunk_length_samples - vocals_chunk.shape[1])))
            if other_chunk.shape[1] < chunk_length_samples:
                other_chunk = cp.pad(other_chunk, ((0, 0), (0, chunk_length_samples - other_chunk.shape[1])))
            
            processed_v, processed_o = process_chunk(
                vocals_chunk, other_chunk, additional_vocals_segments, additional_others_segments, sr
            )
            processed_vocals_chunks.append(processed_v)
            processed_other_chunks.append(processed_o)
        
        # Combine and save
        final_vocals, final_other = combine_chunks(processed_vocals_chunks, processed_other_chunks, desired_length_samples)
        
        song_name = random_song_name()
        song_dir = os.path.join(output_dir, song_name)
        os.makedirs(song_dir, exist_ok=True)
        
        # Convert CuPy to NumPy before saving
        sf.write(os.path.join(song_dir, "vocals.wav"), cp.asnumpy(final_vocals).T, sr)
        sf.write(os.path.join(song_dir, "other.wav"), cp.asnumpy(final_other).T, sr)
        
        # Cleanup GPU memory
        del final_vocals, final_other
        cp.get_default_memory_pool().free_all_blocks()
        
        return f"Successfully created combination: {vocals_path} + {other_path} -> {song_name}"
    
    except Exception as e:
        return f"Error processing {vocals_path} and {other_path}: {e}"

def main():
    vocals_files, other_files = get_all_files(dataset_path)
    
    # Prepare combinations data
    combinations_data = []
    for _ in range(num_combinations):
        vocals_path = random.choice(vocals_files)
        other_path = random.choice(other_files)
        additional_vocals = random.sample(vocals_files, 3)
        additional_others = random.sample(other_files, 3)
        combinations_data.append((vocals_path, other_path, output_path, additional_vocals, additional_others))
    
    # Process in batches
    with tqdm(total=num_combinations, desc="Processing combinations") as pbar:
        for i in range(0, num_combinations, batch_size):
            batch = combinations_data[i:i+batch_size]
            
            # Process batch in parallel
            with ProcessPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(create_combination, data) for data in batch]
                
                for future in as_completed(futures):
                    result = future.result()
                    print(result)
                    pbar.update(1)

if __name__ == "__main__":
    main()