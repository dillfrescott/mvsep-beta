from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import re
import signal
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from adam_atan2_pytorch import AdamAtan2
except ImportError:
    AdamAtan2 = None

try:
    from flash_attn import flash_attn_func as external_flash_attn_func
except (ImportError, OSError) as error:
    external_flash_attn_func = None
    FLASH_ATTN_IMPORT_ERROR: Exception | None = error
else:
    FLASH_ATTN_IMPORT_ERROR = None


@torch.compiler.disable(
    recursive=True,
    reason="flash-attn custom-op fake strides differ from its CUDA output",
)
def run_external_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    """Run fused FlashAttention outside AOTAutograd's fake-layout graph."""
    if external_flash_attn_func is None:
        raise RuntimeError("External flash-attn is unavailable.")
    return external_flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=False,
    )


STEMS = ("vocals", "other")
AUDIO_EXTENSIONS = (".wav", ".flac")
TRAINING_LR = 1e-4


def _environment_path_contains(variable: str, filename: str) -> bool:
    return any(
        (Path(directory) / filename).is_file()
        for directory in os.environ.get(variable, "").split(os.pathsep)
        if directory
    )


def configure_windows_compile_environment() -> Path | None:
    """Initialize MSVC variables needed by TorchInductor when launched normally."""
    if os.name != "nt" or _environment_path_contains("INCLUDE", "omp.h"):
        return None

    candidates: list[Path] = []
    program_files_x86 = Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    )
    vswhere = program_files_x86 / "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.is_file():
        result = subprocess.run(
            (
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        installation = result.stdout.strip()
        if result.returncode == 0 and installation:
            candidates.append(Path(installation) / "Common7/Tools/VsDevCmd.bat")

    for environment_name, fallback in (
        ("ProgramFiles", r"C:\Program Files"),
        ("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ):
        visual_studio_root = (
            Path(os.environ.get(environment_name, fallback)) / "Microsoft Visual Studio"
        )
        if visual_studio_root.is_dir():
            candidates.extend(
                visual_studio_root.glob("*/*/Common7/Tools/VsDevCmd.bat")
            )

    attempted: list[str] = []
    failures: list[str] = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        attempted.append(str(candidate))
        # Pass a complete command line so cmd.exe receives the nested quotes
        # around installations below "Program Files" unchanged.
        command = (
            f'cmd.exe /d /s /c ""{candidate}" -arch=x64 -host_arch=x64 '
            '>nul && set"'
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failure = (result.stderr or result.stdout).strip().splitlines()
            failures.append(failure[-1] if failure else f"exit code {result.returncode}")
            continue
        environment = dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line and not line.startswith("=")
        )
        os.environ.update(environment)
        if _environment_path_contains("INCLUDE", "omp.h"):
            return candidate

    detail = f" Tried: {', '.join(attempted)}." if attempted else ""
    if failures:
        detail += f" Last error: {failures[-1]}"
    raise RuntimeError(
        "--compile on Windows requires the MSVC C++ build tools, but their "
        "environment could not be initialized. Install the Visual Studio "
        f"'Desktop development with C++' workload or omit --compile.{detail}"
    )


def configure_torchinductor_cache() -> Path:
    """Avoid stale graphs after attention custom-op layout changes."""
    configured = os.environ.get("MVSEP_TORCHINDUCTOR_CACHE_DIR")
    cache_dir = (
        Path(configured)
        if configured
        else Path(tempfile.gettempdir()) / "torchinductor_mvsep_submodules_v4"
    )
    # Override a legacy/global TORCHINDUCTOR_CACHE_DIR: it may contain an
    # attention graph whose custom-op fake strides predate this script's
    # contiguous FlashAttention boundary.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    return cache_dir


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    sample_rate: int = 44_100
    n_fft: int = 4096
    hop_length: int = 1024
    win_length: int = 4096
    audio_channels: int = 2
    num_stems: int = len(STEMS)
    num_bands: int = 124
    dim: int = 384
    depth: int = 12
    refine_depth: int = 2
    heads: int = 8
    ff_mult: float = 8.0 / 3.0
    dropout: float = 0.0
    drop_path: float = 0.05
    local_kernel: int = 7
    layer_scale_init: float = 0.1
    refine_mask_scale: float = 0.05
    use_checkpoint: bool = True
    mixture_consistency: bool = True
    architecture: str = "bs_roformer_124"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.n_fft <= 0 or self.n_fft % 2 != 0:
            raise ValueError("n_fft must be a positive even integer.")
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError("win_length must be in the range [1, n_fft].")
        if not 0 < self.hop_length <= self.win_length:
            raise ValueError("hop_length must be in the range [1, win_length].")
        if self.audio_channels != 2:
            raise ValueError("This trainer currently requires stereo audio_channels=2.")
        if self.num_stems != len(STEMS):
            raise ValueError(
                f"num_stems must match STEMS ({len(STEMS)}), got {self.num_stems}."
            )
        if self.num_bands <= 1:
            raise ValueError("num_bands must be greater than one.")
        if self.dim <= 0 or self.depth <= 0 or self.heads <= 0:
            raise ValueError("dim, depth, and heads must be positive.")
        if self.refine_depth < 0:
            raise ValueError("refine_depth cannot be negative.")
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads.")
        if (self.dim // self.heads) % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE.")
        if self.ff_mult <= 0.0:
            raise ValueError("ff_mult must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.drop_path < 1.0:
            raise ValueError("drop_path must be in [0, 1).")
        if self.local_kernel <= 0 or self.local_kernel % 2 == 0:
            raise ValueError("local_kernel must be a positive odd integer.")
        if self.layer_scale_init < 0.0:
            raise ValueError("layer_scale_init must be non-negative.")
        if self.refine_mask_scale < 0.0:
            raise ValueError("refine_mask_scale must be non-negative.")
        if self.architecture != "bs_roformer_124":
            raise ValueError(
                f"Unsupported architecture {self.architecture!r}; expected "
                "bs_roformer_124."
            )


@dataclass
class LossConfig:
    waveform_weight: float = 0.5
    main_stft_weight: float = 1.0
    mrstft_weight: float = 0.5
    mask_weight: float = 0.05
    sdr_weight: float = 0.10
    midside_weight: float = 0.05
    stage1_weight: float = 0.15
    silence_weight: float = 0.05

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "").replace("._orig_mod", "")
        cleaned[key] = value
    return cleaned


@dataclass(frozen=True)
class StateLoadReport:
    matched: int
    incoming: int
    expected: int
    skipped: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_exact(self) -> bool:
        return (
            self.matched == self.expected
            and self.incoming == self.expected
            and not self.skipped
            and not self.missing
        )


def load_matching_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> StateLoadReport:
    """Load keys whose names and shapes match and report exact coverage.

    Partial loading is useful for transfer learning, but it must never be
    mistaken for an exact continuation checkpoint.
    """
    current = module.state_dict()
    incoming = clean_state_dict(state_dict)
    matched: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in incoming.items():
        if key in current and current[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    missing = [key for key in current if key not in matched]
    module.load_state_dict(matched, strict=False)
    return StateLoadReport(
        matched=len(matched),
        incoming=len(incoming),
        expected=len(current),
        skipped=tuple(skipped),
        missing=tuple(missing),
    )


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)



@dataclass(frozen=True)
class BandDefinition:
    """One disjoint frequency interval owned by exactly one band token."""

    start: int
    end: int
    weights: torch.Tensor

    @property
    def width(self) -> int:
        return self.end - self.start


def build_bs_bands(
    n_fft: int,
    num_bands: int,
) -> list[BandDefinition]:
    """Build disjoint BS-RoFormer bands with exact one-owner bin coverage.

    The default 4096-FFT / 124-band layout is a doubled-resolution version of
    the widely used handcrafted 62-band BS-RoFormer partition. The original
    1025-bin layout has widths::

        24 x 2, 12 x 4, 8 x 12, 8 x 24, 8 x 48, 128, 129

    At 4096 FFT resolution there are 2049 real-spectrum bins. Repeating each
    original interval twice gives 124 bands while retaining dense low/mid
    frequency resolution; the final interval is shortened by one bin.

    For non-default FFT/band counts, a deterministic quadratic boundary layout
    supplies many narrow low-frequency bands and progressively wider upper
    bands. In all cases every FFT bin is present exactly once, with no overlap,
    averaging, duplication, or gaps.
    """
    freq_bins = n_fft // 2 + 1
    if num_bands <= 1:
        raise ValueError("num_bands must be greater than one.")
    if num_bands > freq_bins:
        raise ValueError(
            f"Cannot create {num_bands} non-empty bands from {freq_bins} bins."
        )

    if n_fft == 4096 and num_bands == 124:
        base_62 = (
            [2] * 24
            + [4] * 12
            + [12] * 8
            + [24] * 8
            + [48] * 8
            + [128, 129]
        )
        widths = [width for width in base_62 for _ in range(2)]
        # Repetition covers 2050 bins, while a real 4096-point STFT has 2049.
        widths[-1] -= 1
    else:
        positions = torch.linspace(0.0, 1.0, num_bands + 1)
        boundaries = torch.round(positions.square() * freq_bins).long()
        boundaries[0] = 0
        boundaries[-1] = freq_bins
        for index in range(1, num_bands):
            minimum = int(boundaries[index - 1]) + 1
            maximum = freq_bins - (num_bands - index)
            boundaries[index] = boundaries[index].clamp(min=minimum, max=maximum)
        widths = [
            int(boundaries[index + 1] - boundaries[index])
            for index in range(num_bands)
        ]

    if len(widths) != num_bands or sum(widths) != freq_bins:
        raise RuntimeError(
            f"Invalid BS layout: {len(widths)} bands cover {sum(widths)} bins; "
            f"expected {num_bands} bands covering {freq_bins} bins."
        )
    if any(width <= 0 for width in widths):
        raise RuntimeError("BS layout contains an empty band.")

    definitions: list[BandDefinition] = []
    coverage = torch.zeros(freq_bins, dtype=torch.int64)
    start = 0
    for width in widths:
        end = start + width
        definitions.append(
            BandDefinition(
                start=start,
                end=end,
                weights=torch.ones(width, dtype=torch.float32),
            )
        )
        coverage[start:end] += 1
        start = end

    if not torch.all(coverage == 1):
        raise RuntimeError("BS bands must cover every FFT bin exactly once.")
    return definitions

def make_stft(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """STFT for tensors shaped [..., samples]."""
    original_shape = audio.shape[:-1]
    audio_flat = audio.reshape(-1, audio.shape[-1]).float()
    spec = torch.stft(
        audio_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.reshape(*original_shape, spec.shape[-2], spec.shape[-1])


def make_istft(
    spec: torch.Tensor,
    length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """ISTFT for tensors shaped [..., frequency, frames]."""
    original_shape = spec.shape[:-2]
    spec_flat = spec.reshape(-1, spec.shape[-2], spec.shape[-1]).to(torch.complex64)
    audio = torch.istft(
        spec_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )
    return audio.reshape(*original_shape, length)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------



class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.dropout(self.out_proj(F.silu(gate) * value))


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class GatedRoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        use_value_residual: bool = True,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Model dimension must be divisible by the number of heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.use_value_residual = use_value_residual
        self.attention_backend = "fused"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.head_gate = nn.Linear(dim, heads)
        self.value_mix = nn.Linear(dim, heads) if use_value_residual else None
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        value_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # RMSNorm accumulates in FP32 under CUDA autocast. Preserve that
        # numerical behavior, then restore the projection dtype so fused
        # FP16/BF16 attention kernels remain eligible and Q/K/V agree.
        attention_dtype = qkv.dtype
        q = self.q_norm(q).to(dtype=attention_dtype).transpose(1, 2)
        k = self.k_norm(k).to(dtype=attention_dtype).transpose(1, 2)
        v = v.transpose(1, 2)
        original_v = v

        if value_residual is not None and self.value_mix is not None:
            mix = torch.sigmoid(self.value_mix(x)).transpose(1, 2).unsqueeze(-1)
            v = torch.lerp(v, value_residual, mix)

        cos, sin = self.rope(length, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attention_dropout = self.dropout if self.training else 0.0
        use_external_flash = (
            external_flash_attn_func is not None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
            and self.attention_backend in ("fused", "flash")
        )
        if use_external_flash:
            # flash-attn's fake implementation models its output with
            # empty_like(q), while the CUDA kernel returns contiguous BLHD.
            # Materialize BLHD inputs so fake and runtime strides agree when
            # the encoder's axial input itself is non-contiguous.
            flash_q = q.transpose(1, 2).contiguous()
            flash_k = k.transpose(1, 2).contiguous()
            flash_v = v.transpose(1, 2).contiguous()
            out = run_external_flash_attention(
                flash_q,
                flash_k,
                flash_v,
                attention_dropout,
            ).transpose(1, 2)
        else:
            if q.device.type != "cuda" or self.attention_backend == "auto":
                attention_context = contextlib.nullcontext()
            elif self.attention_backend == "fused":
                attention_context = sdpa_kernel(
                    [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                    ],
                    set_priority=True,
                )
            elif self.attention_backend == "flash":
                attention_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            else:
                attention_context = sdpa_kernel(SDPBackend.MATH)
            with attention_context:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=attention_dropout,
                    is_causal=False,
                )

        gates = torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
        out = out * gates
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.out_dropout(self.out_proj(out)), original_v


class AxisConvModule(nn.Module):
    """Conformer-style local sequence mixer used on both time and frequency axes."""

    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = self.depthwise(value.transpose(1, 2))
        value = self.norm(value).transpose(1, 2)
        value = F.silu(value) * torch.sigmoid(gate)
        return self.dropout(self.out_proj(value))


class TransformerUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: float,
        dropout: float,
        drop_path: float,
        local_kernel: int,
        layer_scale_init: float,
    ):
        super().__init__()
        hidden_dim = round_up_to_multiple(round(dim * ff_mult), 64)

        self.attn_norm = nn.RMSNorm(dim)
        self.attn = GatedRoPEAttention(dim, heads, dropout=dropout)
        self.conv_norm = nn.RMSNorm(dim)
        self.conv = AxisConvModule(dim, local_kernel, dropout)
        self.ff_norm = nn.RMSNorm(dim)
        self.ff = SwiGLU(dim, hidden_dim, dropout=dropout)
        self.attn_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.conv_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.ff_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        value_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, original_v = self.attn(self.attn_norm(x), value_residual)
        x = x + self.drop_path(attn_out * self.attn_scale)
        x = x + self.drop_path(self.conv(self.conv_norm(x)) * self.conv_scale)
        x = x + self.drop_path(self.ff(self.ff_norm(x)) * self.ff_scale)
        return x, original_v


class DualPathEncoder(nn.Module):
    def __init__(self, config: ModelConfig, depth: int | None = None):
        super().__init__()
        self.depth = config.depth if depth is None else depth
        if self.depth <= 0:
            raise ValueError("DualPathEncoder depth must be positive.")

        time_layers: list[TransformerUnit] = []
        freq_layers: list[TransformerUnit] = []
        for index in range(self.depth):
            probability = config.drop_path * (index + 1) / self.depth
            unit_kwargs = dict(
                dim=config.dim,
                heads=config.heads,
                ff_mult=config.ff_mult,
                dropout=config.dropout,
                drop_path=probability,
                local_kernel=config.local_kernel,
                layer_scale_init=config.layer_scale_init,
            )
            time_layers.append(TransformerUnit(**unit_kwargs))
            freq_layers.append(TransformerUnit(**unit_kwargs))
        self.time_layers = nn.ModuleList(time_layers)
        self.freq_layers = nn.ModuleList(freq_layers)
        self.output_norm = nn.RMSNorm(config.dim)
        self.use_checkpoint = config.use_checkpoint

    def compile_layers(
        self,
        mode: str = "default",
        recompile_limit: int = 8,
    ) -> None:
        for unit in (*self.time_layers, *self.freq_layers):
            # Keep attention eager: this nightly AOTAutograd build assigns
            # unstable fake strides to saved FlashAttention and Q/K RMSNorm
            # tensors. Compile only the pure local-convolution and feed-forward
            # paths, which still benefit from fused pointwise kernels.
            unit.conv.compile(
                mode=mode,
                recompile_limit=recompile_limit,
                isolate_recompiles=True,
            )
            unit.ff.compile(
                mode=mode,
                recompile_limit=recompile_limit,
                isolate_recompiles=True,
            )

    @staticmethod
    def _run_unit(
        unit: TransformerUnit,
        x: torch.Tensor,
        value_residual: torch.Tensor | None,
        use_checkpoint: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not use_checkpoint:
            return unit(x, value_residual)
        if value_residual is None:
            return checkpoint(lambda z: unit(z, None), x, use_reentrant=False)
        return checkpoint(unit, x, value_residual, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, bands, dim = x.shape
        time_value_residual: torch.Tensor | None = None
        freq_value_residual: torch.Tensor | None = None
        should_checkpoint = self.use_checkpoint and self.training

        for time_layer, freq_layer in zip(self.time_layers, self.freq_layers):
            time_x = x.permute(0, 2, 1, 3).reshape(batch * bands, frames, dim)
            time_x, first_time_values = self._run_unit(
                time_layer, time_x, time_value_residual, should_checkpoint
            )
            if time_value_residual is None:
                time_value_residual = first_time_values
            x = time_x.reshape(batch, bands, frames, dim).permute(0, 2, 1, 3)

            freq_x = x.reshape(batch * frames, bands, dim)
            freq_x, first_freq_values = self._run_unit(
                freq_layer, freq_x, freq_value_residual, should_checkpoint
            )
            if freq_value_residual is None:
                freq_value_residual = first_freq_values
            x = freq_x.reshape(batch, frames, bands, dim)

        return self.output_norm(x)


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def round_up_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive.")
    return int(math.ceil(value / multiple) * multiple)


class BandInputGroup(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[BandDefinition],
        band_ids: Sequence[int],
        bucket_width: int,
        input_channels: int,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.bucket_width = bucket_width
        self.feature_width = bucket_width * input_channels * 2
        self.num_group_bands = len(band_ids)

        self.register_buffer(
            "band_ids", torch.tensor(band_ids, dtype=torch.long), persistent=False
        )
        freq_indices = torch.zeros(self.num_group_bands, bucket_width, dtype=torch.long)
        band_weight = torch.zeros(self.num_group_bands, bucket_width)
        for local_index, band_id in enumerate(band_ids):
            definition = bands[band_id]
            width = definition.width
            freq_indices[local_index, :width] = torch.arange(
                definition.start, definition.end
            )
            band_weight[local_index, :width] = definition.weights

        feature_weight = (
            band_weight.sqrt()[:, :, None, None]
            .expand(-1, -1, input_channels, 2)
            .reshape(self.num_group_bands, self.feature_width)
        )
        effective_counts = (band_weight.sum(dim=-1) * input_channels * 2).clamp_min(1.0)
        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("feature_weight", feature_weight, persistent=False)
        self.register_buffer("effective_counts", effective_counts, persistent=False)

        self.gamma = nn.Parameter(torch.ones(self.num_group_bands, self.feature_width))
        self.weight = nn.Parameter(
            torch.empty(self.num_group_bands, self.feature_width, config.dim)
        )
        self.bias = nn.Parameter(torch.zeros(self.num_group_bands, config.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight)
            for band in range(self.num_group_bands):
                valid = int((self.feature_weight[band] > 0).sum().item())
                bound = math.sqrt(6.0 / max(1, valid + self.weight.shape[-1]))
                self.weight[band, :valid].uniform_(-bound, bound)
            self.gamma.masked_fill_(self.feature_weight == 0, 0.0)

    def forward(self, real_imag: torch.Tensor) -> torch.Tensor:
        # real_imag: [B, T, F, input_channels, 2]
        gathered = real_imag[:, :, self.freq_indices]
        features = gathered.reshape(
            gathered.shape[0],
            gathered.shape[1],
            self.num_group_bands,
            self.feature_width,
        )
        features = features * self.feature_weight[None, None]
        mean_square = features.square().sum(dim=-1, keepdim=True)
        mean_square = mean_square / self.effective_counts[None, None, :, None]
        features = features * torch.rsqrt(mean_square + 1e-5)
        features = features * self.gamma[None, None]
        return torch.einsum("btni,nid->btnd", features, self.weight) + self.bias


class BandSplit(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[BandDefinition],
        input_channels: int,
    ):
        super().__init__()
        self.num_bands = len(bands)
        self.input_channels = input_channels
        grouped_ids: dict[int, list[int]] = {}
        for band_id, definition in enumerate(bands):
            bucket = next_power_of_two(definition.width)
            grouped_ids.setdefault(bucket, []).append(band_id)
        self.groups = nn.ModuleList(
            BandInputGroup(config, bands, ids, bucket, input_channels)
            for bucket, ids in sorted(grouped_ids.items())
        )

    def forward_real(self, real_imag: torch.Tensor) -> torch.Tensor:
        # real_imag: [B, input_channels, F, T, 2]
        if real_imag.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} complex input channels, "
                f"got {real_imag.shape[1]}."
            )
        real_imag = real_imag.permute(0, 3, 2, 1, 4)
        output = real_imag.new_zeros(
            real_imag.shape[0],
            real_imag.shape[1],
            self.num_bands,
            self.groups[0].weight.shape[-1],
        )
        for group in self.groups:
            output = output.index_copy(2, group.band_ids, group(real_imag))
        return output

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.forward_real(torch.view_as_real(spec.to(torch.complex64)))


class BandMaskGroup(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[BandDefinition],
        band_ids: Sequence[int],
        bucket_width: int,
    ):
        super().__init__()
        self.num_stems = config.num_stems
        self.audio_channels = config.audio_channels
        self.bucket_width = bucket_width
        self.feature_width = bucket_width * config.audio_channels * 2
        self.num_group_bands = len(band_ids)

        self.register_buffer(
            "band_ids", torch.tensor(band_ids, dtype=torch.long), persistent=False
        )
        freq_indices = torch.zeros(self.num_group_bands, bucket_width, dtype=torch.long)
        synthesis_weight = torch.zeros(self.num_group_bands, bucket_width)
        for local_index, band_id in enumerate(band_ids):
            definition = bands[band_id]
            width = definition.width
            freq_indices[local_index, :width] = torch.arange(
                definition.start, definition.end
            )
            synthesis_weight[local_index, :width] = definition.weights
        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("synthesis_weight", synthesis_weight, persistent=False)

        output_width = config.num_stems * self.feature_width
        self.output_weight = nn.Parameter(
            torch.empty(self.num_group_bands, config.dim, output_width)
        )
        self.output_bias = nn.Parameter(torch.zeros(self.num_group_bands, output_width))
        nn.init.normal_(self.output_weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.einsum("btnd,ndq->btnq", x, self.output_weight)
        raw = raw + self.output_bias[None, None]
        raw = raw.reshape(
            x.shape[0],
            x.shape[1],
            self.num_group_bands,
            self.num_stems,
            self.bucket_width,
            self.audio_channels,
            2,
        )
        raw = raw * self.synthesis_weight[None, None, :, None, :, None, None]
        source = raw.permute(0, 3, 5, 1, 2, 4, 6).reshape(
            x.shape[0],
            self.num_stems,
            self.audio_channels,
            x.shape[1],
            self.num_group_bands * self.bucket_width,
            2,
        )
        return source, self.freq_indices.reshape(-1)


class BandMaskEstimator(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[BandDefinition],
        *,
        add_partition_bias: bool,
        initial_scale: float,
    ):
        super().__init__()
        self.num_stems = config.num_stems
        self.audio_channels = config.audio_channels
        self.freq_bins = config.n_fft // 2 + 1
        self.num_bands = len(bands)
        self.add_partition_bias = add_partition_bias

        grouped_ids: dict[int, list[int]] = {}
        for band_id, definition in enumerate(bands):
            bucket = next_power_of_two(definition.width)
            grouped_ids.setdefault(bucket, []).append(band_id)
        self.groups = nn.ModuleList(
            BandMaskGroup(config, bands, ids, bucket)
            for bucket, ids in sorted(grouped_ids.items())
        )

        coverage = torch.zeros(self.freq_bins, dtype=torch.int64)
        for definition in bands:
            coverage[definition.start : definition.end] += 1
        if not torch.all(coverage == 1):
            raise ValueError(
                "BandMaskEstimator requires disjoint one-owner frequency coverage."
            )

        hidden_dim = round_up_to_multiple(config.dim * 2.0, 64)
        self.norm = nn.RMSNorm(config.dim)
        self.shared_mlp = SwiGLU(config.dim, hidden_dim, dropout=config.dropout)
        self.mask_residual_scale = nn.Parameter(torch.tensor(initial_scale))

    def forward_real(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.shared_mlp(self.norm(x))
        output = x.new_zeros(
            x.shape[0],
            self.num_stems,
            self.audio_channels,
            x.shape[1],
            self.freq_bins,
            2,
        )
        for group in self.groups:
            group_x = x.index_select(2, group.band_ids)
            source, flat_indices = group(group_x)
            scatter_index = flat_indices.view(1, 1, 1, 1, -1, 1).expand(
                x.shape[0],
                self.num_stems,
                self.audio_channels,
                x.shape[1],
                -1,
                2,
            )
            output.scatter_add_(dim=4, index=scatter_index, src=source)

        output = output.permute(0, 1, 2, 4, 3, 5).contiguous().float()
        output = output * self.mask_residual_scale.float()
        if self.add_partition_bias:
            output = output + output.new_tensor((1.0 / self.num_stems, 0.0))
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(self.forward_real(x))


class GatedTokenFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(dim * 2)
        self.in_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, base: torch.Tensor, refinement: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(self.norm(torch.cat((base, refinement), dim=-1))).chunk(
            2, dim=-1
        )
        update = self.out_proj(torch.sigmoid(gate) * F.silu(value))
        return base + self.scale * update


class BSRoFormerSeparator(nn.Module):
    """Disjoint BS-RoFormer with local mixing and two-stage mask refinement."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bands = build_bs_bands(
            config.n_fft,
            config.num_bands,
        )
        self.band_split = BandSplit(
            config,
            self.bands,
            input_channels=config.audio_channels,
        )
        self.encoder = DualPathEncoder(config, depth=config.depth)
        self.mask_estimator = BandMaskEstimator(
            config,
            self.bands,
            add_partition_bias=True,
            initial_scale=0.1,
        )

        if config.refine_depth > 0:
            self.refine_split: BandSplit | None = BandSplit(
                config,
                self.bands,
                input_channels=config.num_stems * config.audio_channels,
            )
            self.refine_fusion: GatedTokenFusion | None = GatedTokenFusion(config.dim)
            self.refine_encoder: DualPathEncoder | None = DualPathEncoder(
                config, depth=config.refine_depth
            )
            self.refine_mask_estimator: BandMaskEstimator | None = BandMaskEstimator(
                config,
                self.bands,
                add_partition_bias=False,
                initial_scale=config.refine_mask_scale,
            )
        else:
            self.refine_split = None
            self.refine_fusion = None
            self.refine_encoder = None
            self.refine_mask_estimator = None

    def compile_layers(
        self,
        mode: str = "default",
        recompile_limit: int = 8,
    ) -> int:
        self.encoder.compile_layers(mode=mode, recompile_limit=recompile_limit)
        count = len(self.encoder.time_layers) + len(self.encoder.freq_layers)
        if self.refine_encoder is not None:
            self.refine_encoder.compile_layers(
                mode=mode,
                recompile_limit=recompile_limit,
            )
            count += len(self.refine_encoder.time_layers) + len(
                self.refine_encoder.freq_layers
            )
        return count

    def forward_real_stages(
        self,
        mixture_real_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.band_split.forward_real(mixture_real_imag)
        encoded = self.encoder(tokens)
        stage1 = self.mask_estimator.forward_real(encoded)

        if self.refine_encoder is None:
            return stage1, stage1
        assert self.refine_split is not None
        assert self.refine_fusion is not None
        assert self.refine_mask_estimator is not None

        # Refine from the first-stage estimated spectra rather than raw masks.
        # This suppresses meaningless mask values in near-silent mixture bins and
        # gives the second stage a physically grounded representation to repair.
        mixture = mixture_real_imag[:, None]
        mix_real, mix_imag = mixture.unbind(dim=-1)
        mask_real, mask_imag = stage1.unbind(dim=-1)
        estimate_real = mask_real * mix_real - mask_imag * mix_imag
        estimate_imag = mask_real * mix_imag + mask_imag * mix_real
        stage1_estimates = torch.stack((estimate_real, estimate_imag), dim=-1)

        batch, stems, channels, freq, frames, two = stage1_estimates.shape
        stage1_channels = stage1_estimates.reshape(
            batch, stems * channels, freq, frames, two
        )
        refinement_tokens = self.refine_split.forward_real(stage1_channels)
        refinement_tokens = self.refine_fusion(encoded, refinement_tokens)
        refinement_tokens = self.refine_encoder(refinement_tokens)
        correction = self.refine_mask_estimator.forward_real(refinement_tokens)
        return stage1 + correction, stage1

    def forward_real(self, mixture_real_imag: torch.Tensor) -> torch.Tensor:
        final, _ = self.forward_real_stages(mixture_real_imag)
        return final

    def forward_stages(
        self,
        mixture_spec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixture_real_imag = torch.view_as_real(mixture_spec.to(torch.complex64))
        final, stage1 = self.forward_real_stages(mixture_real_imag)
        return torch.view_as_complex(final), torch.view_as_complex(stage1)

    def forward(self, mixture_spec: torch.Tensor) -> torch.Tensor:
        final, _ = self.forward_stages(mixture_spec)
        return final

    def estimate_specs(
        self,
        mixture_spec: torch.Tensor,
        mixture_consistency: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = self(mixture_spec)
        estimates = masks * mixture_spec[:, None]
        use_consistency = (
            self.config.mixture_consistency
            if mixture_consistency is None
            else mixture_consistency
        )
        if use_consistency:
            residual = mixture_spec - estimates.sum(dim=1)
            power = estimates.abs().square().clamp_min(1e-8)
            weights = power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)
            estimates = estimates + weights * residual[:, None]
        return estimates, masks

# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------



class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        resolutions: Sequence[tuple[int, int, int]] = (
            (2048, 512, 2048),
            (1024, 256, 1024),
            (512, 128, 512),
            (256, 64, 256),
        ),
        activity_threshold: float = 1e-4,
        compression: float = 0.3,
    ):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.activity_threshold = activity_threshold
        self.compression = compression
        for index, (_, _, win_length) in enumerate(self.resolutions):
            self.register_buffer(
                f"window_{index}", torch.hann_window(win_length), persistent=False
            )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = prediction.reshape(-1, prediction.shape[-1]).float()
        target_flat = target.reshape(-1, target.shape[-1]).float()
        active_targets = (
            target_flat.square().mean(dim=1).sqrt() >= self.activity_threshold
        )
        total = pred_flat.new_tensor(0.0)

        for index, (n_fft, hop_length, win_length) in enumerate(self.resolutions):
            window = getattr(self, f"window_{index}")
            # One larger cuFFT launch is faster than separate prediction/target
            # launches and has the same gradients (the target half is detached).
            combined_spec = torch.stft(
                torch.cat((pred_flat, target_flat), dim=0),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
            pred_spec, target_spec = combined_spec.split(
                (pred_flat.shape[0], target_flat.shape[0]), dim=0
            )
            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()

            diff_norm = torch.linalg.vector_norm(
                (pred_mag - target_mag).flatten(1), dim=1
            )
            target_norm = torch.linalg.vector_norm(target_mag.flatten(1), dim=1)
            active_diff_norm = diff_norm[active_targets]
            active_target_norm = target_norm[active_targets]
            spectral_convergence = (
                active_diff_norm / active_target_norm.clamp_min(1e-6)
            ).sum() / active_targets.count_nonzero().clamp_min(1)

            log_magnitude = F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag))
            complex_normalizer = target_mag.mean().detach().clamp_min(1e-4)
            complex_loss = (pred_spec - target_spec).abs().mean() / complex_normalizer

            pred_compressed = pred_spec / pred_mag.clamp_min(1e-8)
            pred_compressed = pred_compressed * pred_mag.clamp_min(1e-8).pow(
                self.compression
            )
            target_compressed = target_spec / target_mag.clamp_min(1e-8)
            target_compressed = target_compressed * target_mag.clamp_min(1e-8).pow(
                self.compression
            )
            compressed_loss = F.l1_loss(
                torch.view_as_real(pred_compressed),
                torch.view_as_real(target_compressed),
            )
            total = total + (
                spectral_convergence
                + log_magnitude
                + 0.25 * complex_loss
                + 0.25 * compressed_loss
            )

        return total / len(self.resolutions)


def normalized_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).abs().mean(dim=-1)
    scale = target.abs().mean(dim=-1).clamp_min(1e-4)
    return (error / scale).mean()


def scale_dependent_sdr_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    error_power = (prediction - target).square().mean(dim=(-2, -1))
    target_power = target.square().mean(dim=(-2, -1))
    valid = target_power > 1e-7
    ratio_db = 10.0 * torch.log10(
        (target_power + 1e-8) / (error_power + 1e-8)
    )
    ratio_db = ratio_db.clamp(-30.0, 30.0)
    if valid.any():
        return -ratio_db[valid].mean()
    return prediction.new_tensor(0.0)


def mid_side(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mid = (audio[..., 0, :] + audio[..., 1, :]) * 0.5
    side = (audio[..., 0, :] - audio[..., 1, :]) * 0.5
    return mid, side


def compressed_complex_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    compression: float = 0.3,
) -> torch.Tensor:
    pred_mag = prediction.abs().clamp_min(1e-8)
    target_mag = target.abs().clamp_min(1e-8)
    pred = prediction / pred_mag * pred_mag.pow(compression)
    true = target / target_mag * target_mag.pow(compression)
    return F.l1_loss(torch.view_as_real(pred), torch.view_as_real(true))


class SeparationLoss(nn.Module):
    def __init__(self, model_config: ModelConfig, loss_config: LossConfig):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.mrstft = MultiResolutionSTFTLoss()
        self.register_buffer(
            "window", torch.hann_window(model_config.win_length), persistent=False
        )

    def forward(
        self,
        model: BSRoFormerSeparator,
        mixture_spec: torch.Tensor,
        target_audio: torch.Tensor,
        target_specs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks, stage1_masks = model.forward_stages(mixture_spec)
        estimates = masks * mixture_spec[:, None]
        stage1_estimates = stage1_masks * mixture_spec[:, None]
        if self.model_config.mixture_consistency:
            residual = mixture_spec - estimates.sum(dim=1)
            power = estimates.abs().square().clamp_min(1e-8)
            weights = power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)
            estimates = estimates + weights * residual[:, None]

        pred_audio = make_istft(
            estimates,
            length=target_audio.shape[-1],
            n_fft=self.model_config.n_fft,
            hop_length=self.model_config.hop_length,
            win_length=self.model_config.win_length,
            window=self.window,
        )
        if target_specs is None:
            target_specs = make_stft(
                target_audio,
                n_fft=self.model_config.n_fft,
                hop_length=self.model_config.hop_length,
                win_length=self.model_config.win_length,
                window=self.window,
            )

        wave_loss = normalized_l1(pred_audio, target_audio)
        mrstft_loss = self.mrstft(pred_audio, target_audio)

        target_mag = target_specs.abs()
        spec_normalizer = target_mag.mean().detach().clamp_min(1e-4)
        main_complex = (estimates - target_specs).abs().mean() / spec_normalizer
        main_logmag = F.l1_loss(torch.log1p(estimates.abs()), torch.log1p(target_mag))
        main_compressed = compressed_complex_loss(estimates, target_specs)
        main_stft_loss = main_complex + main_logmag + 0.5 * main_compressed

        stage1_complex = (stage1_estimates - target_specs).abs().mean() / spec_normalizer
        stage1_logmag = F.l1_loss(
            torch.log1p(stage1_estimates.abs()), torch.log1p(target_mag)
        )
        stage1_loss = stage1_complex + stage1_logmag

        mix_power = mixture_spec.abs().square()
        ideal_masks = (
            target_specs * mixture_spec[:, None].conj()
            / (mix_power[:, None] + 1e-5)
        )
        ideal_mag = ideal_masks.abs().clamp_max(8.0)
        ideal_masks = torch.polar(ideal_mag, torch.angle(ideal_masks))
        tf_weight = mixture_spec.abs()
        tf_weight = tf_weight / tf_weight.mean(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-4)
        tf_weight = tf_weight.clamp(max=10.0)
        mask_loss = ((masks - ideal_masks).abs() * tf_weight[:, None]).mean()
        mask_loss = mask_loss + 0.25 * (
            (stage1_masks - ideal_masks).abs() * tf_weight[:, None]
        ).mean()

        sdr_loss = scale_dependent_sdr_loss(pred_audio, target_audio)
        pred_mid, pred_side = mid_side(pred_audio)
        true_mid, true_side = mid_side(target_audio)
        midside_loss = 0.5 * (
            normalized_l1(pred_mid, true_mid)
            + normalized_l1(pred_side, true_side)
        )

        target_rms = target_audio.square().mean(dim=(-2, -1)).sqrt()
        silent = target_rms < 1e-4
        if silent.any():
            mixture_rms = target_audio.sum(dim=1).square().mean(dim=(-2, -1)).sqrt()
            leakage = pred_audio.square().mean(dim=(-2, -1)).sqrt()
            silence_loss = (
                leakage / mixture_rms[:, None].clamp_min(1e-4)
            )[silent].mean()
        else:
            silence_loss = pred_audio.new_tensor(0.0)

        cfg = self.loss_config
        total = (
            cfg.waveform_weight * wave_loss
            + cfg.main_stft_weight * main_stft_loss
            + cfg.mrstft_weight * mrstft_loss
            + cfg.mask_weight * mask_loss
            + cfg.sdr_weight * sdr_loss
            + cfg.midside_weight * midside_loss
            + cfg.stage1_weight * stage1_loss
            + cfg.silence_weight * silence_loss
        )
        metrics = {
            "wave": wave_loss.detach(),
            "main_stft": main_stft_loss.detach(),
            "mrstft": mrstft_loss.detach(),
            "mask": mask_loss.detach(),
            "sdr_loss": sdr_loss.detach(),
            "midside": midside_loss.detach(),
            "stage1": stage1_loss.detach(),
            "silence": silence_loss.detach(),
        }
        return total, metrics

# -----------------------------------------------------------------------------
# Dataset and augmentation
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioInfo:
    path: str
    frames: int
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


class StemDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 44_100,
        segment_samples: int = 352_800,
        virtual_size: int = 50_000,
        remix_probability: float = 0.5,
        min_activity_rms: float = 1e-4,
        eq_probability: float = 0.25,
        stem_dropout_probability: float = 0.05,
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.segment_seconds = segment_samples / sample_rate
        self.virtual_size = virtual_size
        self.remix_probability = remix_probability
        self.min_activity_rms = min_activity_rms
        self.eq_probability = eq_probability
        self.stem_dropout_probability = stem_dropout_probability
        self.tracks: list[dict[str, AudioInfo]] = []

        track_dirs = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]
        print("Scanning track metadata...")
        for track_dir in tqdm(track_dirs, desc="Caching tracks"):
            track: dict[str, AudioInfo] = {}
            for stem in STEMS:
                path = self._find_audio_file(track_dir, stem)
                if path is None:
                    break
                info = sf.info(path)
                track[stem] = AudioInfo(path, info.frames, info.samplerate)
            if len(track) == len(STEMS):
                self.tracks.append(track)

        if not self.tracks:
            raise RuntimeError(f"No complete {STEMS} tracks found under {root_dir!r}.")
        print(f"Cached {len(self.tracks)} complete tracks.")

    @staticmethod
    def _find_audio_file(directory: str, stem: str) -> str | None:
        by_lower_name = {name.lower(): name for name in os.listdir(directory)}
        for extension in AUDIO_EXTENSIONS:
            candidate = f"{stem}{extension}"
            actual = by_lower_name.get(candidate.lower())
            if actual is not None:
                return os.path.join(directory, actual)
        return None

    def __len__(self) -> int:
        return self.virtual_size

    def _load_segment(self, info: AudioInfo, start_seconds: float) -> torch.Tensor:
        source_start = int(round(start_seconds * info.sample_rate))
        source_frames = int(math.ceil(self.segment_seconds * info.sample_rate))
        source_start = max(0, min(source_start, max(0, info.frames - 1)))
        audio_np, source_sr = sf.read(
            info.path,
            start=source_start,
            frames=source_frames,
            dtype="float32",
            always_2d=True,
        )
        audio = torch.from_numpy(audio_np.T)
        audio = torch.nan_to_num(audio)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        if source_sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_sr, self.sample_rate)

        if audio.shape[-1] < self.segment_samples:
            audio = F.pad(audio, (0, self.segment_samples - audio.shape[-1]))
        else:
            audio = audio[..., : self.segment_samples]
        return audio.contiguous()

    def _random_start(self, info: AudioInfo) -> float:
        return random.uniform(0.0, max(0.0, info.duration - self.segment_seconds))

    def _sample_targets(self) -> torch.Tensor:
        targets: list[torch.Tensor] = []
        if random.random() < self.remix_probability:
            for stem in STEMS:
                track = random.choice(self.tracks)
                info = track[stem]
                targets.append(self._load_segment(info, self._random_start(info)))
        else:
            track = random.choice(self.tracks)
            common_duration = min(track[stem].duration for stem in STEMS)
            start = random.uniform(0.0, max(0.0, common_duration - self.segment_seconds))
            for stem in STEMS:
                targets.append(self._load_segment(track[stem], start))
        return torch.stack(targets)

    def _random_eq(self, audio: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.eq_probability:
            return audio
        output = audio
        for _ in range(random.randint(1, 2)):
            low = math.log(60.0)
            high = math.log(min(16_000.0, self.sample_rate * 0.45))
            center = math.exp(random.uniform(low, high))
            gain = random.uniform(-4.5, 4.5)
            q = math.exp(random.uniform(math.log(0.45), math.log(2.0)))
            output = torchaudio.functional.equalizer_biquad(
                output,
                self.sample_rate,
                center_freq=center,
                gain=gain,
                Q=q,
            )
        return torch.nan_to_num(output)

    def _augment(self, targets: torch.Tensor) -> torch.Tensor:
        gains_db = torch.empty(targets.shape[0]).uniform_(-8.0, 4.0)
        gains = torch.pow(10.0, gains_db / 20.0).view(-1, 1, 1)
        targets = targets * gains

        for stem_index in range(targets.shape[0]):
            targets[stem_index] = self._random_eq(targets[stem_index])
            if random.random() < 0.5:
                targets[stem_index] = -targets[stem_index]

            width = random.uniform(0.65, 1.35)
            mid = (targets[stem_index, 0] + targets[stem_index, 1]) * 0.5
            side = (targets[stem_index, 0] - targets[stem_index, 1]) * 0.5 * width
            targets[stem_index, 0] = mid + side
            targets[stem_index, 1] = mid - side

            balance_db = random.uniform(-2.0, 2.0)
            targets[stem_index, 0] *= db_to_gain(balance_db)
            targets[stem_index, 1] *= db_to_gain(-balance_db)

        # Pure-vocal and pure-instrumental examples strongly reduce leakage in
        # silent regions without changing the dataset's file contract.
        if random.random() < self.stem_dropout_probability:
            targets[random.randrange(targets.shape[0])].zero_()

        if random.random() < 0.5:
            targets = targets.flip(dims=(1,))

        global_gain = db_to_gain(random.uniform(-4.0, 3.0))
        targets = targets * global_gain
        peak = targets.sum(dim=0).abs().amax()
        if peak > 1.0:
            targets = targets * (0.98 / peak)
        return targets.contiguous()

    def __getitem__(self, _: int) -> torch.Tensor:
        last_error: Exception | None = None
        for _attempt in range(20):
            try:
                targets = self._augment(self._sample_targets())
                mixture = targets.sum(dim=0)
                if mixture.square().mean().sqrt() < self.min_activity_rms:
                    continue
                # Training derives the mixture spectrum by summing the target
                # spectra, so returning a duplicate mixture would only add CPU
                # collation, pinned-memory, and host-to-device transfer work.
                return targets
            except Exception as error:
                last_error = error
        raise RuntimeError(f"Unable to load a valid training example: {last_error}")


# -----------------------------------------------------------------------------
# EMA, optimizer, checkpointing
# -----------------------------------------------------------------------------


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.model = model
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        self.updates += 1
        # Warm the decay up so early validation is not dominated by the random
        # initialization. It approaches the requested long-horizon decay.
        warm_decay = (1.0 + self.updates) / (10.0 + self.updates)
        decay = min(self.decay, warm_decay)
        trainable = [
            (self.shadow[name], parameter.detach())
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            shadow_tensors, parameter_tensors = zip(*trainable)
            torch._foreach_lerp_(shadow_tensors, parameter_tensors, 1.0 - decay)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        if self.backup:
            raise RuntimeError("EMA shadow weights are already applied.")
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}

    @contextlib.contextmanager
    def average_parameters(self):
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        updates: int = 0,
    ) -> StateLoadReport:
        report = load_matching_state_dict(_EMAStateView(self.shadow), state_dict)
        self.updates = max(0, int(updates))
        return report


class _EMAStateView(nn.Module):
    """Minimal adapter that lets EMA tensors use the strict state-load report."""

    def __init__(self, shadow: dict[str, torch.Tensor]):
        super().__init__()
        self.shadow = shadow

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        del args, kwargs
        return self.shadow.copy()

    def load_state_dict(self, state_dict, strict=False):  # type: ignore[override]
        del strict
        for name, value in state_dict.items():
            self.shadow[name].copy_(value)
        return None


def build_optimizer(
    model: nn.Module,
    weight_decay: float,
    optimizer_name: str,
    fused: bool = False,
) -> torch.optim.Optimizer:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        use_no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in lowered
            or name.endswith(".gamma")
            or name.endswith("_scale")
        )
        (no_decay_params if use_no_decay else decay_params).append(parameter)

    parameter_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if optimizer_name == "atan2" and AdamAtan2 is not None:
        return AdamAtan2(parameter_groups, lr=TRAINING_LR)
    if optimizer_name == "atan2" and AdamAtan2 is None:
        print("adam_atan2_pytorch is unavailable; falling back to AdamW.")
    if fused:
        print("Optimizer: fused CUDA AdamW.")
    return torch.optim.AdamW(
        parameter_groups,
        lr=TRAINING_LR,
        betas=(0.9, 0.95),
        fused=fused,
    )


def find_latest_checkpoint(folder: str = "ckpts") -> str | None:
    paths = list(Path(folder).glob("checkpoint_step_*.pt"))
    if not paths:
        return None

    def step(path: Path) -> int:
        match = re.search(r"step_(\d+)", path.name)
        return int(match.group(1)) if match else 0

    return str(max(paths, key=step))


def find_latest_compatible_checkpoint(
    config: ModelConfig,
    folder: str = "ckpts",
) -> str | None:
    paths = list(Path(folder).glob("checkpoint_step_*.pt"))

    def step(path: Path) -> int:
        match = re.search(r"step_(\d+)", path.name)
        return int(match.group(1)) if match else 0

    for path in sorted(paths, key=step, reverse=True):
        try:
            checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            print(f"Ignoring unreadable checkpoint {path}: {error}")
            continue
        if checkpoint_data.get("model_config") == asdict(config):
            return str(path)
    return None


def find_best_checkpoint(folder: str = "best_ckpts") -> str | None:
    scored: list[tuple[float, Path]] = []
    for path in Path(folder).glob("*.pt"):
        match = re.search(r"sdr_(-?\d+(?:\.\d+)?)\.pt$", path.name)
        if match:
            scored.append((float(match.group(1)), path))
    return str(max(scored, key=lambda item: item[0])[1]) if scored else None


def save_checkpoint(
    path: str,
    model: BSRoFormerSeparator,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    best_sdr: float,
    avg_loss: float,
) -> None:
    payload = {
        "step": step,
        "model_state_dict": clean_state_dict(model.state_dict()),
        "ema_state_dict": clean_state_dict(ema.state_dict()),
        "ema_updates": ema.updates,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_sdr": best_sdr,
        "avg_loss": avg_loss,
        "stems": STEMS,
        "model_config": asdict(model.config),
        "checkpoint_format_version": 2,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def prune_old_checkpoints(folder: str, keep: int = 3) -> None:
    paths = sorted(Path(folder).glob("*.pt"), key=lambda path: path.stat().st_mtime)
    for path in paths[:-keep]:
        path.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Inference and validation
# -----------------------------------------------------------------------------


def crossfade_window(
    length: int,
    overlap: int,
    fade_in: bool,
    fade_out: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    window = torch.ones(length, device=device, dtype=dtype)
    fade_length = min(overlap, length)
    if fade_length <= 0:
        return window
    phase = torch.linspace(0.0, math.pi / 2.0, fade_length, device=device, dtype=dtype)
    fade = torch.sin(phase).square()
    if fade_in:
        window[:fade_length] = fade
    if fade_out:
        window[-fade_length:] = fade.flip(0)
    return window


def chunk_starts(total_length: int, chunk_size: int, overlap: int) -> list[int]:
    if total_length <= chunk_size:
        return [0]
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")
    starts = list(range(0, total_length - chunk_size + 1, step))
    final_start = total_length - chunk_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def separate_tensor(
    model: BSRoFormerSeparator,
    mixture: torch.Tensor,
    chunk_size: int,
    overlap: int,
    device: torch.device,
    precision: str = "bf16",
    show_progress: bool = False,
) -> list[torch.Tensor]:
    if mixture.ndim != 2:
        raise ValueError("Mixture must be [channels, samples].")
    if mixture.shape[0] == 1:
        mixture = mixture.repeat(2, 1)
    if mixture.shape[0] != model.config.audio_channels:
        raise ValueError(
            f"Expected {model.config.audio_channels} channels, got {mixture.shape[0]}."
        )

    mixture = mixture.to(device=device, dtype=torch.float32)
    total_length = mixture.shape[-1]
    starts = chunk_starts(total_length, chunk_size, overlap)
    output = torch.zeros(
        model.config.num_stems,
        model.config.audio_channels,
        total_length,
        device=device,
    )
    weight_sum = torch.zeros(total_length, device=device)
    stft_window = torch.hann_window(model.config.win_length, device=device)

    iterator: Iterable[int] = starts
    if show_progress:
        iterator = tqdm(starts, desc="Separating", leave=False)

    for start in iterator:
        usable = min(chunk_size, total_length - start)
        chunk = mixture[:, start : start + usable]
        if usable < chunk_size:
            pad = chunk_size - usable
            if usable > 1:
                reflect = min(pad, usable - 1)
                chunk = F.pad(chunk, (0, reflect), mode="reflect")
                if reflect < pad:
                    chunk = F.pad(chunk, (0, pad - reflect))
            else:
                chunk = F.pad(chunk, (0, pad))

        spec = make_stft(
            chunk.unsqueeze(0),
            n_fft=model.config.n_fft,
            hop_length=model.config.hop_length,
            win_length=model.config.win_length,
            window=stft_window,
        )
        with autocast_context(device, precision):
            estimated_specs, _ = model.estimate_specs(spec)
        estimated = make_istft(
            estimated_specs,
            length=chunk_size,
            n_fft=model.config.n_fft,
            hop_length=model.config.hop_length,
            win_length=model.config.win_length,
            window=stft_window,
        ).squeeze(0)

        is_first = start == 0
        is_last = start + chunk_size >= total_length
        window = crossfade_window(
            chunk_size,
            overlap,
            fade_in=not is_first,
            fade_out=not is_last,
            device=device,
            dtype=estimated.dtype,
        )[:usable]
        output[..., start : start + usable] += estimated[..., :usable] * window
        weight_sum[start : start + usable] += window

    output = output / weight_sum.clamp_min(1e-8)

    # Enforce exact waveform mixture consistency after overlap-add. Do not clamp;
    # clipping predictions changes SDR and should only happen at file export time.
    residual = mixture - output.sum(dim=0)
    power = output.abs().square().clamp_min(1e-8)
    weights = power / power.sum(dim=0, keepdim=True).clamp_min(1e-8)
    output = output + weights * residual.unsqueeze(0)
    return [output[index] for index in range(model.config.num_stems)]


def calculate_sdr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Scale-dependent global SDR over all channels and samples."""
    error = prediction - target
    signal_power = target.double().square().sum()
    error_power = error.double().square().sum()
    sdr = 10.0 * torch.log10((signal_power + 1e-12) / (error_power + 1e-12))
    return float(sdr)


def find_stem_file(directory: str, stem: str) -> str:
    by_lower_name = {name.lower(): name for name in os.listdir(directory)}
    for extension in AUDIO_EXTENSIONS:
        candidate = f"{stem}{extension}"
        actual = by_lower_name.get(candidate.lower())
        if actual is not None:
            return os.path.join(directory, actual)
    raise FileNotFoundError(f"Could not find an exact {stem} WAV/FLAC in {directory!r}.")


@torch.inference_mode()
def validate(
    model: BSRoFormerSeparator,
    test_dir: str,
    device: torch.device,
    chunk_size: int,
    overlap: int,
    precision: str,
) -> tuple[list[float], float | None]:
    model.eval()
    track_dirs = [
        os.path.join(test_dir, name)
        for name in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, name))
    ] if os.path.isdir(test_dir) else []
    if not track_dirs:
        print(f"No validation tracks found under {test_dir!r}.")
        return [0.0 for _ in STEMS], None

    totals = [0.0 for _ in STEMS]
    count = 0
    progress = tqdm(track_dirs, desc="Validating", leave=False)
    for track_dir in progress:
        try:
            targets: list[torch.Tensor] = []
            for stem in STEMS:
                path = find_stem_file(track_dir, stem)
                audio_np, sample_rate = sf.read(
                    path,
                    dtype="float32",
                    always_2d=True,
                )
                if sample_rate != model.config.sample_rate:
                    raise ValueError(
                        f"Validation file {path} is {sample_rate} Hz, expected "
                        f"{model.config.sample_rate} Hz."
                    )
                audio = torch.from_numpy(audio_np.T)
                audio = torch.nan_to_num(audio)
                if audio.shape[0] == 1:
                    audio = audio.repeat(2, 1)
                targets.append(audio[:2])

            length = min(target.shape[-1] for target in targets)
            targets = [target[..., :length].to(device) for target in targets]
            mixture = torch.stack(targets).sum(dim=0)
            predictions = separate_tensor(
                model,
                mixture,
                chunk_size=chunk_size,
                overlap=overlap,
                device=device,
                precision=precision,
                show_progress=False,
            )
            scores = [calculate_sdr(pred, target) for pred, target in zip(predictions, targets)]
            for index, score in enumerate(scores):
                totals[index] += score
            count += 1
            progress.set_postfix_str(
                " | ".join(
                    f"{stem}: {score:.3f}" for stem, score in zip(STEMS, scores)
                )
            )
        except Exception as error:
            print(f"\nSkipping {track_dir}: {error}")

    if count == 0:
        return [0.0 for _ in STEMS], None
    averages = [total / count for total in totals]
    return averages, sum(averages) / len(averages)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(
    model: BSRoFormerSeparator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_module: SeparationLoss,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: str | None,
) -> None:
    model.to(device)
    loss_module.to(device)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    step = 0
    best_sdr = -float("inf")
    avg_loss = 0.0
    checkpoint_data: dict | None = None
    exact_continuation = False

    if checkpoint_path:
        checkpoint_data = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_state = checkpoint_data.get("model_state_dict", checkpoint_data)
        report = load_matching_state_dict(model, model_state)
        print(
            f"Loaded {report.matched}/{report.expected} model tensors from "
            f"{checkpoint_path} ({len(report.skipped)} skipped, "
            f"{len(report.missing)} missing)."
        )
        saved_config = checkpoint_data.get("model_config")
        exact_continuation = saved_config == asdict(model.config) and report.is_exact

    # Initialize after model loading so a transfer checkpoint also seeds EMA.
    ema = EMA(model, decay=args.ema_decay)

    if checkpoint_data is not None and exact_continuation:
        if args.reset_optimizer:
            print(
                "Loaded exact raw weights, but --reset_optimizer starts fresh "
                "optimizer, EMA, and global-step timelines."
            )
        else:
            if "ema_state_dict" in checkpoint_data:
                ema_report = ema.load_state_dict(
                    checkpoint_data["ema_state_dict"],
                    updates=int(checkpoint_data.get("ema_updates", 0)),
                )
                if not ema_report.is_exact:
                    raise RuntimeError(
                        "Exact model checkpoint has an incomplete EMA state: "
                        f"{ema_report.matched}/{ema_report.expected} tensors matched."
                    )
                print(f"Loaded complete EMA state at update {ema.updates}.")

            step = int(checkpoint_data.get("step", 0))
            best_sdr = float(checkpoint_data.get("best_sdr", best_sdr))
            avg_loss = float(checkpoint_data.get("avg_loss", 0.0))
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            else:
                raise RuntimeError("Continuation checkpoint has no optimizer state.")
            # Optimizer checkpoints include their current learning rates. Override
            # older scheduled values so resumed runs remain locked to TRAINING_LR.
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = TRAINING_LR
                parameter_group["initial_lr"] = TRAINING_LR
            if "scaler_state_dict" in checkpoint_data:
                scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
            print(f"Resuming at optimizer step {step}.")
    elif checkpoint_data is not None:
        print(
            "Checkpoint is not an exact continuation. Using shape-matched weights "
            "as a transfer initialization with a fresh optimizer, EMA "
            "timeline, and global step."
        )

    print(f"Learning rate locked at {TRAINING_LR:.1e}.")

    if args.compile:
        compiler_environment = configure_windows_compile_environment()
        if compiler_environment is not None:
            print(
                "Initialized the Windows C++ build environment from "
                f"{compiler_environment}."
            )
        configure_torchinductor_cache()
        compiled_units = model.compile_layers(mode="default", recompile_limit=8)
        print(
            f"Compiled local-convolution and feed-forward paths in {compiled_units} "
            "axial transformer units; attention and checkpoint boundaries remain eager."
        )

    train_model = model

    stft_window = torch.hann_window(model.config.win_length, device=device)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        initial=step,
        dynamic_ncols=True,
        disable=None,
        mininterval=0.5,
        bar_format="{desc} | {n_fmt} steps [{elapsed}, {rate_fmt}{postfix}]",
    )
    data_iterator = iter(dataloader)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        progress.write(
            "Stop requested; finishing the current optimizer step and saving. "
            "Press Ctrl+C again to force an immediate exit."
        )

    previous_sigint_handler = signal.signal(signal.SIGINT, request_stop)

    while not stop_requested and (args.max_steps <= 0 or step < args.max_steps):
        model.train()
        accumulated_loss_tensor = torch.zeros((), device=device)
        latest_metrics: dict[str, torch.Tensor] = {}
        completed_accumulation = True

        for _micro_step in range(args.grad_accumulation):
            try:
                target_audio = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                target_audio = next(data_iterator)

            target_audio = target_audio.to(device, non_blocking=True)
            # STFT is linear, so transforming the stems once and summing their
            # spectra avoids a separate mixture STFT and reuses the target STFT
            # needed by the loss.
            target_specs = make_stft(
                target_audio,
                n_fft=model.config.n_fft,
                hop_length=model.config.hop_length,
                win_length=model.config.win_length,
                window=stft_window,
            )
            mixture_spec = target_specs.sum(dim=1)

            with autocast_context(device, args.precision):
                loss, latest_metrics = loss_module(
                    train_model,  # type: ignore[arg-type]
                    mixture_spec,
                    target_audio,
                    target_specs=target_specs,
                )
                scaled_loss = loss / args.grad_accumulation

            if not torch.isfinite(loss):
                print(f"Non-finite loss at step {step}; discarding gradients.")
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss_tensor.zero_()
                completed_accumulation = False
                break

            scaler.scale(scaled_loss).backward()
            accumulated_loss_tensor.add_(loss.detach())

        if not completed_accumulation:
            continue

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=False
        )
        if not torch.isfinite(grad_norm):
            print(f"Non-finite gradient norm at step {step}; skipping optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update()

        step += 1
        accumulated_loss = float(
            accumulated_loss_tensor.div_(args.grad_accumulation)
        )
        avg_loss = (
            accumulated_loss
            if step == 1
            else 0.995 * avg_loss + 0.005 * accumulated_loss
        )
        current_lr = optimizer.param_groups[0]["lr"]
        progress.set_description(
            f"Step {step} | loss {accumulated_loss:.4f} | avg {avg_loss:.4f} "
            f"| lr {current_lr:.2e} | grad {float(grad_norm):.2f} "
            f"| best {best_sdr:.4f}",
            refresh=False,
        )
        if latest_metrics:
            wave, main_stft, mrstft = torch.stack(
                (
                    latest_metrics["wave"],
                    latest_metrics["main_stft"],
                    latest_metrics["mrstft"],
                )
            ).float().cpu().tolist()
            progress.set_postfix(
                wave=f"{wave:.3f}",
                stft=f"{main_stft:.3f}",
                mr=f"{mrstft:.3f}",
                refresh=False,
            )
        progress.update(1)

        if step % args.checkpoint_steps == 0:
            with ema.average_parameters():
                # The transformer units are compiled for training with gradients
                # enabled.  Validation runs under inference_mode, which requires
                # different Dynamo guards and can exhaust the shared recompile
                # cache for TransformerUnit.forward.  Keep infrequent validation
                # eager so it cannot evict or disable the compiled training path.
                compiler_context = (
                    torch.compiler.set_stance("force_eager")
                    if args.compile
                    else contextlib.nullcontext()
                )
                with compiler_context:
                    stem_scores, combined_sdr = validate(
                        model,
                        args.test_dir,
                        device,
                        chunk_size=args.segment_samples,
                        overlap=args.inference_overlap,
                        precision=args.precision,
                    )

            improved = combined_sdr is not None and combined_sdr > best_sdr
            if combined_sdr is None:
                print("\nValidation produced no valid tracks; best SDR was not changed.")
            else:
                score_text = ", ".join(
                    f"{stem}: {score:.4f} dB"
                    for stem, score in zip(STEMS, stem_scores)
                )
                print(
                    f"\nValidation step {step} (EMA, track-mean global SDR): "
                    f"{score_text}, combined: {combined_sdr:.4f} dB"
                )
                if improved:
                    best_sdr = combined_sdr

            # Save after validation so latest checkpoints never contain a stale
            # pre-validation best score.
            regular_path = f"ckpts/checkpoint_step_{step}.pt"
            save_checkpoint(
                regular_path,
                model,
                ema,
                optimizer,
                scaler,
                step,
                best_sdr,
                avg_loss,
            )
            prune_old_checkpoints("ckpts", keep=3)

            if improved and combined_sdr is not None:
                best_path = (
                    f"best_ckpts/checkpoint_step_{step}_sdr_{combined_sdr:.4f}.pt"
                )
                save_checkpoint(
                    best_path,
                    model,
                    ema,
                    optimizer,
                    scaler,
                    step,
                    best_sdr,
                    avg_loss,
                )
                for old_path in Path("best_ckpts").glob("*.pt"):
                    if old_path != Path(best_path):
                        old_path.unlink(missing_ok=True)
                print(f"New best checkpoint: {best_path}\n")
            elif combined_sdr is not None:
                print(f"Best combined SDR remains {best_sdr:.4f} dB.\n")

    progress.close()
    signal.signal(signal.SIGINT, previous_sigint_handler)
    if step > 0:
        stopped_path = f"ckpts/checkpoint_step_{step}.pt"
        save_checkpoint(
            stopped_path,
            model,
            ema,
            optimizer,
            scaler,
            step,
            best_sdr,
            avg_loss,
        )
        prune_old_checkpoints("ckpts", keep=3)
        reason = "reached max_steps" if args.max_steps > 0 and step >= args.max_steps else "stopped cleanly"
        print(f"Training {reason}; checkpoint saved to {stopped_path}.")


# -----------------------------------------------------------------------------
# Command line entry point
# -----------------------------------------------------------------------------


def model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        audio_channels=2,
        num_stems=len(STEMS),
        num_bands=args.num_bands,
        dim=args.model_dim,
        depth=args.depth,
        refine_depth=args.refine_depth,
        heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        drop_path=args.drop_path,
        local_kernel=args.local_kernel,
        layer_scale_init=args.layer_scale_init,
        refine_mask_scale=args.refine_mask_scale,
        use_checkpoint=args.ckpt,
        mixture_consistency=not args.disable_mixture_consistency,
        architecture="bs_roformer_124",
    )


def inspect_checkpoint_config(
    checkpoint_path: str,
    fallback: ModelConfig,
) -> ModelConfig:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_stems = tuple(checkpoint_data.get("stems", STEMS))
    if saved_stems != STEMS:
        raise ValueError(
            f"Checkpoint stems {saved_stems} do not match this script's STEMS {STEMS}."
        )
    config_data = checkpoint_data.get("model_config")
    if not config_data:
        return fallback
    valid_fields = ModelConfig.__dataclass_fields__.keys()
    filtered = {key: value for key, value in config_data.items() if key in valid_fields}
    return ModelConfig(**filtered)


def load_inference_weights(
    model: BSRoFormerSeparator,
    checkpoint_path: str,
) -> None:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint_data.get("ema_state_dict") or checkpoint_data.get("model_state_dict")
    if state is None:
        state = checkpoint_data
    report = load_matching_state_dict(model, state)
    if not report.is_exact:
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"loaded {report.matched}/{report.expected} required tensors; "
            f"{len(report.skipped)} incoming tensors were incompatible and "
            f"{len(report.missing)} required tensors were missing."
        )
    print(f"Loaded EMA/model weights from {checkpoint_path}.")


def read_input_audio(path: str, sample_rate: int) -> torch.Tensor:
    audio_np, source_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio_np.T)
    audio = torch.nan_to_num(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if source_sr != sample_rate:
        audio = torchaudio.functional.resample(audio, source_sr, sample_rate)
    return audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "4090-tuned disjoint BS-RoFormer with local convolution and "
            "two-stage complex-mask refinement"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--infer", action="store_true")

    parser.add_argument("--data_dir", type=str, default="train")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new run instead of auto-resuming the latest compatible checkpoint.",
    )

    parser.add_argument("--sample_rate", type=int, default=44_100)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--hop_length", type=int, default=1024)
    parser.add_argument("--win_length", type=int, default=4096)
    parser.add_argument("--segment_seconds", type=float, default=5.0)
    parser.add_argument("--inference_overlap_seconds", type=float, default=2.0)

    parser.add_argument("--num_bands", type=int, default=124)
    parser.add_argument("--model_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--refine_depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_mult", type=float, default=8.0 / 3.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.05)
    parser.add_argument("--local_kernel", type=int, default=7)
    parser.add_argument("--layer_scale_init", type=float, default=0.1)
    parser.add_argument("--refine_mask_scale", type=float, default=0.05)
    parser.add_argument(
        "--ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint transformer activations to fit the default model in 24 GB.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile convolution/FFN submodules (slower startup, faster steady state).",
    )
    parser.add_argument(
        "--attention_backend",
        choices=("fused", "flash", "auto", "math"),
        default="fused",
        help=(
            "CUDA attention backend. 'fused' tries external flash-attn, PyTorch "
            "Flash, then memory-efficient attention with no math fallback."
        ),
    )
    parser.add_argument("--disable_mixture_consistency", action="store_true")

    # A five-second crop with batch 1 sustains >1.3 optimizer steps/s on the
    # target RTX 4090 while retaining the full 93M-parameter architecture.
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--dataset_size", type=int, default=50_000)
    parser.add_argument("--remix_probability", type=float, default=0.5)
    parser.add_argument("--eq_probability", type=float, default=0.25)
    parser.add_argument("--stem_dropout_probability", type=float, default=0.05)
    parser.add_argument("--checkpoint_steps", type=int, default=5_000)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps; 0 trains until stopped.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adamw", "atan2"), default="adamw")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def validate_runtime_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "batch_size",
        "grad_accumulation",
        "dataset_size",
        "checkpoint_steps",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive.")
    if args.max_steps < 0:
        raise ValueError("--max_steps cannot be negative.")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad_clip must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight_decay cannot be negative.")
    if not 0.0 <= args.remix_probability <= 1.0:
        raise ValueError("--remix_probability must be in [0, 1].")
    if not 0.0 <= args.eq_probability <= 1.0:
        raise ValueError("--eq_probability must be in [0, 1].")
    if not 0.0 <= args.stem_dropout_probability <= 1.0:
        raise ValueError("--stem_dropout_probability must be in [0, 1].")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema_decay must be in [0, 1).")
    if args.segment_seconds <= 0.0:
        raise ValueError("--segment_seconds must be positive.")
    if args.inference_overlap_seconds < 0.0:
        raise ValueError("--inference_overlap_seconds cannot be negative.")
    if args.dataset_size < args.batch_size:
        raise ValueError("--dataset_size must be at least --batch_size.")


def main() -> None:
    args = build_parser().parse_args()
    validate_runtime_args(args)
    seed_everything(args.seed)
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("best_ckpts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.segment_samples = int(round(args.segment_seconds * args.sample_rate))
    args.inference_overlap = int(
        round(args.inference_overlap_seconds * args.sample_rate)
    )
    if args.segment_samples < args.win_length:
        raise ValueError(
            "Training/inference segments must contain at least one full STFT window."
        )
    if args.inference_overlap >= args.segment_samples:
        raise ValueError("Inference overlap must be smaller than the segment length.")

    config = model_config_from_args(args)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        if args.train and not args.fresh:
            checkpoint_path = find_latest_compatible_checkpoint(config, "ckpts")
            if checkpoint_path is None and find_latest_checkpoint("ckpts") is not None:
                print(
                    "No compatible auto-resume checkpoint was found. Starting fresh; "
                    "use --checkpoint_path explicitly for shape-matched transfer."
                )
        elif args.infer:
            checkpoint_path = (
                find_latest_checkpoint("ckpts")
                if args.latest
                else find_best_checkpoint("best_ckpts")
            )

    if args.infer and checkpoint_path:
        config = inspect_checkpoint_config(checkpoint_path, config)
        # Keep chunk timing tied to the checkpoint sample rate.
        args.segment_samples = int(round(args.segment_seconds * config.sample_rate))
        args.inference_overlap = int(
            round(args.inference_overlap_seconds * config.sample_rate)
        )
        if args.segment_samples < config.win_length:
            raise ValueError(
                "Inference segment must contain at least one checkpoint STFT window."
            )
        if args.inference_overlap >= args.segment_samples:
            raise ValueError("Inference overlap must be smaller than segment length.")

    if device.type == "cuda" and args.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("CUDA device lacks BF16 support; falling back to FP16.")
            args.precision = "fp16"

    model = BSRoFormerSeparator(config)
    for module in model.modules():
        if isinstance(module, GatedRoPEAttention):
            module.attention_backend = args.attention_backend
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"BS-RoFormer Local+Refine parameters: {parameter_count / 1e6:.2f}M")
    if device.type == "cuda":
        flash_available = getattr(
            torch.backends.cuda, "is_flash_attention_available", lambda: True
        )()
        if args.attention_backend == "flash":
            if external_flash_attn_func is not None:
                print("CUDA attention backend: external flash-attn package (required).")
            elif not flash_available:
                import_detail = (
                    f" Import error: {FLASH_ATTN_IMPORT_ERROR}"
                    if FLASH_ATTN_IMPORT_ERROR is not None
                    else ""
                )
                raise RuntimeError(
                    "Neither the external flash-attn package nor this PyTorch build "
                    f"provides Flash Attention.{import_detail}"
                )
            else:
                print("CUDA attention backend: built-in PyTorch Flash Attention.")
        elif args.attention_backend == "fused":
            if external_flash_attn_func is not None:
                print(
                    "CUDA attention backend: external flash-attn package "
                    "(memory-efficient fallback available)."
                )
            elif flash_available:
                print(
                    "CUDA attention backend: fused (PyTorch Flash, then "
                    "memory-efficient; math disabled)."
                )
            else:
                print(
                    "Flash Attention is unavailable in this PyTorch build; using "
                    "fused memory-efficient attention with math disabled."
                )
        else:
            print(f"CUDA attention backend: {args.attention_backend}.")

    if args.train:
        if checkpoint_path:
            print(f"Checkpoint selected: {checkpoint_path}")
        print(
            "Training throughput: "
            f"batch {args.batch_size} x accumulation {args.grad_accumulation} "
            f"= effective batch {args.batch_size * args.grad_accumulation}; "
            f"{args.segment_seconds * args.batch_size * args.grad_accumulation:.1f} "
            "audio-seconds/step; "
            f"{args.num_workers} workers, prefetch {args.prefetch_factor}."
        )
        dataset = StemDataset(
            root_dir=args.data_dir,
            sample_rate=config.sample_rate,
            segment_samples=args.segment_samples,
            virtual_size=args.dataset_size,
            remix_probability=args.remix_probability,
            eq_probability=args.eq_probability,
            stem_dropout_probability=args.stem_dropout_probability,
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=True,
        )
        optimizer = build_optimizer(
            model,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            fused=device.type == "cuda",
        )
        loss_module = SeparationLoss(config, LossConfig())
        train(
            model,
            dataloader,
            optimizer,
            loss_module,
            device,
            args,
            checkpoint_path,
        )
        return

    if not args.input_file:
        raise ValueError("--input_file is required for inference.")
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint was supplied or found.")

    load_inference_weights(model, checkpoint_path)
    model.to(device).eval()
    mixture = read_input_audio(args.input_file, config.sample_rate)
    predictions = separate_tensor(
        model,
        mixture,
        chunk_size=args.segment_samples,
        overlap=args.inference_overlap,
        device=device,
        precision=args.precision,
        show_progress=True,
    )
    for stem, prediction in zip(STEMS, predictions):
        output_path = os.path.join("outputs", f"{stem}.wav")
        # WAV encoders may clip at export, but internal inference remains unclamped.
        sf.write(output_path, prediction.cpu().numpy().T, config.sample_rate, subtype="FLOAT")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
