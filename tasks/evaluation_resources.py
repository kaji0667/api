"""Shared read-only resource accounting for task evaluation reports."""

import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import psutil
import torch


MB = 1024**2


def synchronize(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def reset_peak_gpu_memory(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
        torch.cuda.synchronize(resolved)


def peak_gpu_memory_mb(device: str | torch.device) -> float | None:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(resolved) / MB)


def process_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / MB)


def latency_summary(values_ms: Iterable[float]) -> dict[str, float | int | None]:
    values = np.asarray(list(values_ms), dtype=np.float64)
    if values.size == 0:
        return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    return {
        "samples": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def model_parameters(*models: torch.nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for model in models for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for model in models for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for model in models for parameter in model.parameters()
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "loaded_parameter_memory_mb": float(parameter_bytes / MB),
    }


def file_size_mb(path: str | Path) -> float | None:
    resolved = Path(path).expanduser().resolve()
    return float(resolved.stat().st_size / MB) if resolved.is_file() else None


def resolve_openclip_weight_file(
    explicit_path: str | Path = "",
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
) -> Path | None:
    """Resolve an explicitly supplied or standard OpenAI OpenCLIP cache artifact without downloading."""
    if explicit_path:
        resolved = Path(explicit_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"OpenCLIP weight file not found: {resolved}")
        return resolved
    if str(pretrained).strip().lower() != "openai":
        return None
    filename = f"{str(model_name).strip().replace('/', '-')}.pt"
    candidates = (
        Path.home() / ".cache" / "clip" / filename,
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename,
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def environment_summary(device: str | torch.device) -> dict[str, object]:
    resolved = torch.device(device)
    return {
        "device": str(resolved),
        "gpu_name": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "process_id": int(psutil.Process().pid),
        "python_executable": sys.executable,
    }
