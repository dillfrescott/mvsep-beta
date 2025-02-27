import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Source and Destination Roots
SRC_ROOT = 'test'
DST_ROOT = 'new_test'

# Ensure destination root exists
os.makedirs(DST_ROOT, exist_ok=True)

def merge_stems(stem_paths):
    """Merge all audio files from given paths into one, preserving stereo."""
    merged_audio = None
    sr = None  # Sample Rate
    
    for filepath in stem_paths:
        audio, current_sr = librosa.load(filepath, sr=None, mono=False)  # Load in stereo
        
        if sr is None:
            sr = current_sr
        else:
            assert sr == current_sr, "Sample rates do not match"
        
        # Ensure audio arrays are the same length; truncate to the shortest if needed
        if merged_audio is None:
            merged_audio = audio
        else:
            min_length = min(merged_audio.shape[1], audio.shape[1])
            merged_audio = merged_audio[:, :min_length] + audio[:, :min_length]
    
    if merged_audio is not None:
        return merged_audio.astype(np.float32), sr
    else:
        return None, None

def process_song_directory(song_dir):
    """Process each song directory, merging drums, bass, and other stems."""
    song_name = os.path.basename(song_dir)
    dst_song_dir = os.path.join(DST_ROOT, song_name)
    os.makedirs(dst_song_dir, exist_ok=True)
    
    # Collect file paths for drums, bass, and other stems
    other_files = [os.path.join(song_dir, filename) for filename in os.listdir(song_dir) if filename in ['bass.wav', 'drums.wav', 'other.wav']]
    vocals_file = os.path.join(song_dir, 'vocals.wav')
    mixture_file = os.path.join(song_dir, 'mixture.wav')
    
    # Process and save other.wav
    if other_files:
        merged_other, sr = merge_stems(other_files)
        if merged_other is not None:
            sf.write(os.path.join(dst_song_dir, 'other.wav'), merged_other.T, sr)  # Transpose for stereo output
    
    # Copy vocals.wav if it exists
    if os.path.exists(vocals_file):
        sf.write(os.path.join(dst_song_dir, 'vocals.wav'), librosa.load(vocals_file, sr=None, mono=False)[0].T, librosa.load(vocals_file, sr=None, mono=False)[1])  # Transpose for stereo output
    
    # Copy mixture.wav if it exists
    if os.path.exists(mixture_file):
        sf.write(os.path.join(dst_song_dir, 'mixture.wav'), librosa.load(mixture_file, sr=None, mono=False)[0].T, librosa.load(mixture_file, sr=None, mono=False)[1])  # Transpose for stereo output

def main():
    # Get list of all song directories
    song_dirs = [os.path.join(SRC_ROOT, song_dir) for song_dir in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, song_dir))]
    
    # Progress bar for overall processing of song directories
    for song_dir in tqdm(song_dirs, desc="Processing songs"):
        process_song_directory(song_dir)

if __name__ == "__main__":
    main()
