import os
import random
import cupy as cp
import numpy as np
import string
import librosa
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Paths and parameters
dataset_path = "train"
output_path = "combined_train"  # changed output folder name to reflect non-augmented stems
num_combinations = 4000
batch_size = 10
desired_length_sec = 4 * 60  # 4 minutes
chunk_length_sec = 60        # 1 minute
sr = 44100                   # Sample rate

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

def process_chunk(vocals_chunk, other_chunk, additional_vocals_segments, additional_others_segments, sr):
    # No augmentation is applied; simply return the original chunks.
    return vocals_chunk, other_chunk

def combine_chunks(processed_vocals_chunks, processed_other_chunks, desired_length_samples):
    final_vocals = cp.concatenate(processed_vocals_chunks, axis=1)
    final_other = cp.concatenate(processed_other_chunks, axis=1)
    
    # Trim or pad to the desired length
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
        
        # Function to load, convert, and prepare audio segments
        def load_and_prepare(paths):
            segments = []
            for path in paths:
                # Load audio using librosa (as NumPy)
                audio, _ = librosa.load(path, sr=sr, mono=False, dtype=np.float32)
                if audio.ndim == 1:
                    audio = np.stack([audio, audio])  # Convert mono to stereo
                # Convert to CuPy
                audio = cp.asarray(audio)
                # Pad or trim to the chunk length
                if audio.shape[1] < chunk_length_samples:
                    audio = cp.pad(audio, ((0, 0), (0, chunk_length_samples - audio.shape[1])))
                else:
                    audio = audio[:, :chunk_length_samples]
                segments.append(audio)
            return segments
        
        additional_vocals_segments = load_and_prepare(additional_vocals)
        additional_others_segments = load_and_prepare(additional_others)
        
        # Load main vocals and other stems
        vocals, _ = librosa.load(vocals_path, sr=sr, mono=False, dtype=np.float32)
        if vocals.ndim == 1:
            vocals = np.stack([vocals, vocals])
        vocals = cp.asarray(vocals)
        if vocals.shape[1] < chunk_length_samples:
            vocals = cp.pad(vocals, ((0, 0), (0, chunk_length_samples - vocals.shape[1])))
        
        other, _ = librosa.load(other_path, sr=sr, mono=False, dtype=np.float32)
        if other.ndim == 1:
            other = np.stack([other, other])
        other = cp.asarray(other)
        if other.shape[1] < chunk_length_samples:
            other = cp.pad(other, ((0, 0), (0, chunk_length_samples - other.shape[1])))
        
        # Process chunks sequentially
        num_chunks_needed = int(desired_length_samples // chunk_length_samples)
        processed_vocals_chunks = []
        processed_other_chunks = []
        
        for _ in range(num_chunks_needed):
            # Randomly choose between using the main stem or one of the additional segments
            if random.random() < 0.5:
                vocals_chunk = vocals[:, :chunk_length_samples]
                other_chunk = other[:, :chunk_length_samples]
            else:
                vocals_chunk = random.choice(additional_vocals_segments)
                other_chunk = random.choice(additional_others_segments)
            
            # Ensure each chunk has the proper length
            if vocals_chunk.shape[1] < chunk_length_samples:
                vocals_chunk = cp.pad(vocals_chunk, ((0, 0), (0, chunk_length_samples - vocals_chunk.shape[1])))
            if other_chunk.shape[1] < chunk_length_samples:
                other_chunk = cp.pad(other_chunk, ((0, 0), (0, chunk_length_samples - other_chunk.shape[1])))
            
            processed_v, processed_o = process_chunk(
                vocals_chunk, other_chunk, additional_vocals_segments, additional_others_segments, sr
            )
            processed_vocals_chunks.append(processed_v)
            processed_other_chunks.append(processed_o)
        
        # Combine all processed chunks and save the resulting stems
        final_vocals, final_other = combine_chunks(processed_vocals_chunks, processed_other_chunks, desired_length_samples)
        
        song_name = random_song_name()
        song_dir = os.path.join(output_dir, song_name)
        os.makedirs(song_dir, exist_ok=True)
        
        # Convert CuPy arrays back to NumPy before saving
        sf.write(os.path.join(song_dir, "vocals.wav"), cp.asnumpy(final_vocals).T, sr)
        sf.write(os.path.join(song_dir, "other.wav"), cp.asnumpy(final_other).T, sr)
        
        # Clean up GPU memory
        del final_vocals, final_other
        cp.get_default_memory_pool().free_all_blocks()
        
        return f"Successfully created combination: {vocals_path} + {other_path} -> {song_name}"
    
    except Exception as e:
        return f"Error processing {vocals_path} and {other_path}: {e}"

def main():
    vocals_files, other_files = get_all_files(dataset_path)
    
    # Prepare data for combinations
    combinations_data = []
    for _ in range(num_combinations):
        vocals_path = random.choice(vocals_files)
        other_path = random.choice(other_files)
        additional_vocals = random.sample(vocals_files, 3)
        additional_others = random.sample(other_files, 3)
        combinations_data.append((vocals_path, other_path, output_path, additional_vocals, additional_others))
    
    # Process combinations in batches using parallel processing
    with tqdm(total=num_combinations, desc="Processing combinations") as pbar:
        for i in range(0, num_combinations, batch_size):
            batch = combinations_data[i:i+batch_size]
            with ProcessPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(create_combination, data) for data in batch]
                for future in as_completed(futures):
                    result = future.result()
                    print(result)
                    pbar.update(1)

if __name__ == "__main__":
    main()