#!/usr/bin/env python3
"""Separate multiple songs with one MVSEP model load.

Each input is written to its own directory beneath ``--output-dir``::

    batch_outputs/
        first_song/
            vocals.wav
            other.wav
        second_song/
            vocals.wav
            other.wav

Input files may be supplied directly or discovered in directories. Songs are
processed sequentially so the model only needs to occupy GPU memory once.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_EXTENSIONS = (".aif", ".aiff", ".flac", ".mp3", ".ogg", ".wav")
OUTPUT_FORMATS = {
    "wav": ("WAV", "FLOAT"),
    "flac": ("FLAC", "PCM_24"),
}


@dataclass(frozen=True)
class SongJob:
    input_path: Path
    output_dir: Path


def normalized_extensions(values: Sequence[str]) -> set[str]:
    """Normalize command-line extensions to lowercase values with a dot."""
    extensions = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in values
    }
    if not extensions or "." in extensions:
        raise ValueError("--extensions must contain at least one file extension.")
    return extensions


def discover_inputs(
    inputs: Sequence[Path],
    recursive: bool,
    extensions: set[str],
    excluded_directory: Path | None = None,
) -> list[Path]:
    """Expand files and directories into a unique, deterministic song list."""
    discovered: list[Path] = []
    missing: list[Path] = []
    excluded_directory = (
        excluded_directory.resolve() if excluded_directory is not None else None
    )

    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            # Explicit files are accepted regardless of extension. The audio
            # reader will provide a useful error if their format is unsupported.
            discovered.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            discovered.extend(
                candidate.resolve()
                for candidate in iterator
                if candidate.is_file()
                and candidate.suffix.lower() in extensions
                and (
                    excluded_directory is None
                    or not candidate.resolve().is_relative_to(excluded_directory)
                )
            )
        else:
            missing.append(raw_path)

    if missing:
        raise FileNotFoundError(
            "Input path(s) do not exist: " + ", ".join(str(path) for path in missing)
        )

    unique = sorted(set(discovered), key=lambda path: str(path).casefold())
    if not unique:
        extension_list = ", ".join(sorted(extensions))
        raise ValueError(f"No audio files found (directory extensions: {extension_list}).")
    return unique


def build_jobs(input_paths: Sequence[Path], output_root: Path) -> list[SongJob]:
    """Assign stable output folders, disambiguating duplicate file stems."""
    stem_counts: dict[str, int] = {}
    for path in input_paths:
        key = path.stem.casefold()
        stem_counts[key] = stem_counts.get(key, 0) + 1

    jobs: list[SongJob] = []
    for path in input_paths:
        label = path.stem
        if stem_counts[path.stem.casefold()] > 1:
            digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
            label = f"{label}__{digest}"
        jobs.append(SongJob(path, output_root / label))
    return jobs


def resolve_checkpoint(args: argparse.Namespace, main_dir: Path, mvsep: Any) -> Path:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
    elif args.latest:
        found = mvsep.find_latest_checkpoint(str(main_dir / "ckpts"))
        checkpoint = Path(found).resolve() if found else Path()
    else:
        found = mvsep.find_best_checkpoint(str(main_dir / "best_ckpts"))
        checkpoint = Path(found).resolve() if found else Path()

    if not checkpoint.is_file():
        selection = "latest ckpts checkpoint" if args.latest else "best checkpoint"
        raise FileNotFoundError(
            f"Could not find the {selection}. Pass its path with --checkpoint."
        )
    return checkpoint


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Path]:
    """Import MVSEP and initialize the selected checkpoint once."""
    main_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(main_dir))
    import mvsep  # pylint: disable=import-outside-toplevel

    checkpoint = resolve_checkpoint(args, main_dir, mvsep)
    fallback = mvsep.ModelConfig(use_checkpoint=False)
    config = mvsep.inspect_checkpoint_config(str(checkpoint), fallback)
    config.use_checkpoint = False

    torch = mvsep.torch
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if args.device is not None else default_device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("CUDA device lacks BF16 support; falling back to FP16.")
            args.precision = "fp16"
    elif args.precision != "fp32":
        print(f"{device.type.upper()} inference uses FP32; ignoring --precision={args.precision}.")
        args.precision = "fp32"

    model = mvsep.BSRoFormerSeparator(config)
    for module in model.modules():
        if isinstance(module, mvsep.GatedRoPEAttention):
            module.attention_backend = args.attention_backend
    mvsep.load_inference_weights(model, str(checkpoint))
    model.to(device).eval()
    return mvsep, model, device, checkpoint


def selected_output_paths(job: SongJob, stems: Sequence[str], extension: str) -> list[Path]:
    return [job.output_dir / f"{stem}.{extension}" for stem in stems]


def existing_job_state(job: SongJob, stems: Sequence[str], extension: str) -> str:
    existing = [path.exists() for path in selected_output_paths(job, stems, extension)]
    if all(existing):
        return "complete"
    if any(existing):
        return "partial"
    return "new"


def write_song(
    job: SongJob,
    predictions: Sequence[Any],
    model_stems: Sequence[str],
    selected_stems: Sequence[str],
    sample_rate: int,
    extension: str,
    soundfile: Any,
) -> None:
    """Write all selected stems via temporary files before replacing outputs."""
    format_name, subtype = OUTPUT_FORMATS[extension]
    by_stem = dict(zip(model_stems, predictions))
    job.output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=job.output_dir.parent, prefix=f".{job.output_dir.name}_"
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        for stem in selected_stems:
            prediction = by_stem[stem]
            temporary_path = temporary_dir / f"{stem}.{extension}"
            soundfile.write(
                temporary_path,
                prediction.detach().float().cpu().numpy().T,
                sample_rate,
                format=format_name,
                subtype=subtype,
            )

        job.output_dir.mkdir(parents=True, exist_ok=True)
        for stem in selected_stems:
            os.replace(
                temporary_dir / f"{stem}.{extension}",
                job.output_dir / f"{stem}.{extension}",
            )


def process_jobs(args: argparse.Namespace, jobs: Sequence[SongJob]) -> int:
    pending: list[SongJob] = []
    failures: list[tuple[Path, str]] = []
    for job in jobs:
        state = existing_job_state(job, args.stems, args.format)
        if state == "complete" and not args.overwrite:
            print(f"Skipping complete song: {job.input_path}")
            continue
        if state == "partial" and not args.overwrite:
            outputs = selected_output_paths(job, args.stems, args.format)
            error = (
                "Partial outputs already exist: "
                + ", ".join(str(path) for path in outputs if path.exists())
                + ". Pass --overwrite to replace them."
            )
            print(f"ERROR: {job.input_path}: {error}", file=sys.stderr)
            failures.append((job.input_path, error))
            if args.fail_fast:
                break
            continue
        pending.append(job)

    if failures and args.fail_fast:
        print("Failed to process 1 song.", file=sys.stderr)
        return 1
    if not pending:
        if failures:
            print(f"Failed to process {len(failures)} song(s).", file=sys.stderr)
            return 1
        print("All songs already have complete outputs.")
        return 0

    mvsep, model, device, checkpoint = load_model(args)
    config = model.config
    chunk_size = int(round(args.segment_seconds * config.sample_rate))
    overlap = int(round(args.overlap_seconds * config.sample_rate))
    if chunk_size < config.win_length:
        raise ValueError("--segment-seconds is too short for the checkpoint STFT window.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("--overlap-seconds must be non-negative and shorter than the segment.")
    unknown_stems = sorted(set(args.stems) - set(mvsep.STEMS))
    if unknown_stems:
        raise ValueError(
            f"Checkpoint model has stems {tuple(mvsep.STEMS)!r}; unsupported selection: "
            + ", ".join(unknown_stems)
        )

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}; precision: {args.precision}")
    print(f"Processing {len(pending)} song(s) into {args.output_dir.resolve()}")

    for index, job in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] {job.input_path}")
        audio = None
        predictions = None
        try:
            audio = mvsep.read_input_audio(str(job.input_path), config.sample_rate)
            predictions = mvsep.separate_tensor(
                model,
                audio,
                chunk_size=chunk_size,
                overlap=overlap,
                device=device,
                precision=args.precision,
                show_progress=True,
            )
            for stem, prediction in zip(mvsep.STEMS, predictions):
                if not mvsep.torch.isfinite(prediction).all():
                    raise RuntimeError(f"Non-finite samples in predicted {stem} stem.")
            write_song(
                job,
                predictions,
                mvsep.STEMS,
                args.stems,
                config.sample_rate,
                args.format,
                mvsep.sf,
            )
            print(f"Saved {job.output_dir}")
        except Exception as exc:  # Continue the batch unless fail-fast was requested.
            print(f"ERROR: {exc}", file=sys.stderr)
            failures.append((job.input_path, str(exc)))
            # Release failed tensors before emptying the CUDA allocator cache.
            audio = None
            predictions = None
            if device.type == "cuda":
                mvsep.torch.cuda.empty_cache()
            if args.fail_fast:
                break

    if failures:
        print(f"\nFailed to process {len(failures)} song(s):", file=sys.stderr)
        for path, error in failures:
            print(f"  {path}: {error}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process multiple songs with MVSEP while loading the model only once. "
            "Each song receives its own output directory."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Audio files and/or directories containing audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_outputs"),
        help="Output root (default: batch_outputs).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        metavar="EXT",
        help="Directory-discovery extensions (explicit files are always accepted).",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: highest-SDR checkpoint in best_ckpts).",
    )
    checkpoint.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest ckpts/checkpoint_step_*.pt instead of best_ckpts.",
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=["vocals", "other"],
        metavar="STEM",
        help="Model stems to save (default: vocals other).",
    )
    parser.add_argument(
        "--format",
        choices=tuple(OUTPUT_FORMATS),
        default="wav",
        help="WAV/FLOAT preserves output; FLAC/PCM_24 is smaller but quantized.",
    )
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument(
        "--attention-backend",
        choices=("fused", "flash", "auto", "math"),
        default="fused",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing selected stem files instead of skipping complete songs.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed song instead of continuing the batch.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Show discovered inputs and output directories without loading the model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.segment_seconds <= 0:
        raise ValueError("--segment-seconds must be positive.")
    if args.overlap_seconds < 0:
        raise ValueError("--overlap-seconds cannot be negative.")
    if len(args.stems) != len(set(args.stems)):
        raise ValueError("--stems cannot contain duplicates.")

    extensions = normalized_extensions(args.extensions)
    output_root = args.output_dir.expanduser().resolve()
    input_paths = discover_inputs(
        args.inputs,
        args.recursive,
        extensions,
        excluded_directory=output_root,
    )
    jobs = build_jobs(input_paths, output_root)

    if args.list_only:
        for job in jobs:
            print(f"{job.input_path} -> {job.output_dir}")
        print(f"Discovered {len(jobs)} song(s).")
        return 0
    return process_jobs(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
