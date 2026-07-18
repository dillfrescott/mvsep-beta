<img width="400" alt="mvsep" src="https://github.com/user-attachments/assets/eec7a8a9-8c72-4721-889c-d8103ad7d353" />

### DISCLAIMER:

This repo is probably going to have a lot of changes done to it and some changes may make quality worse until I can decide what works and what doesn't. Please keep that in mind. Use any code at your own risk of wasting time and compute.

### Batch processing

`main/tools/batch_process.py` processes multiple files while loading the model
only once. Pass individual songs, directories, or both:

```bash
cd main
python tools/batch_process.py song1.wav song2.flac --checkpoint path/to/checkpoint.pt
python tools/batch_process.py music/ --recursive --output-dir separated
```

Outputs are stored as `<output-dir>/<song-name>/vocals.wav` and
`<output-dir>/<song-name>/other.wav`. Existing complete songs are skipped so an
interrupted batch can be resumed; pass `--overwrite` to replace them. Use
`--list-only` to preview the discovered songs and output paths without loading
the model.
