import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

def combine_audio_files(file1, file2, output_file):
    # Read the WAV files
    rate1, data1 = wavfile.read(file1)
    rate2, data2 = wavfile.read(file2)
    
    # Ensure the sampling rates are the same
    if rate1 != rate2:
        raise ValueError(f"Sampling rates do not match: {rate1} vs {rate2}")

    # Ensure both files have the same shape; if not, trim to the shorter length
    min_length = min(data1.shape[0], data2.shape[0])
    if data1.shape[0] != data2.shape[0]:
        print(f"Warning: Files {file1} and {file2} have different lengths. Trimming to {min_length} samples.")
        data1 = data1[:min_length]
        data2 = data2[:min_length]

    # Convert data to float for safe averaging and avoid overflow
    data1_float = data1.astype(np.float32)
    data2_float = data2.astype(np.float32)

    # Mix by averaging the two signals
    mixture_float = (data1_float + data2_float) / 2.0

    # Convert back to the original data type (e.g., int16)
    mixture = mixture_float.astype(data1.dtype)

    # Write the resulting mixture file
    wavfile.write(output_file, rate1, mixture)
    print(f"Saved mixture to {output_file}")

def process_dataset(root_dir):
    # Get only the subdirectories (ignore the root directory entry)
    all_dirs = [entry for entry in os.scandir(root_dir) if entry.is_dir()]
    
    # Wrap the directory list with tqdm for progress display
    for subdir in tqdm(all_dirs, desc="Processing directories"):
        subdir_path = subdir.path
        files = os.listdir(subdir_path)
        if 'other.wav' in files and 'vocals.wav' in files:
            other_path = os.path.join(subdir_path, 'other.wav')
            vocals_path = os.path.join(subdir_path, 'vocals.wav')
            mixture_path = os.path.join(subdir_path, 'mixture.wav')
            try:
                combine_audio_files(other_path, vocals_path, mixture_path)
            except Exception as e:
                print(f"Error processing {subdir_path}: {e}")

if __name__ == '__main__':
    # Set the dataset root directory (e.g., "train")
    dataset_dir = 'train'
    process_dataset(dataset_dir)