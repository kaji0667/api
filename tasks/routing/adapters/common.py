"""Shared browser-image and model-resource helpers for task adapters."""

from __future__ import annotations

import base64
import binascii
import gc
import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..config import TaskConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_DECODED_IMAGE_BYTES = 40 * 1024 * 1024


def resolve_checkpoint(config: TaskConfig, environment_name: str) -> Path:
    """Resolve an environment-overridable checkpoint against HFSA-main."""

    configured = os.environ.get(environment_name, config.checkpoint).strip()
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def resolve_project_path(value: str | Path, environment_name: str = "") -> Path:
    """Resolve a project-relative runtime artifact with an optional environment override."""

    configured = os.environ.get(environment_name, str(value)).strip() if environment_name else str(value)
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def decode_data_url(value: Any) -> Image.Image:
    """Decode the bounded browser image payload into an owned RGB PIL image."""

    if not isinstance(value, Mapping):
        raise ValueError("图像输入格式无效，请重新选择本地图像。")
    data_url = value.get("data_url")
    if not isinstance(data_url, str) or not data_url.strip():
        raise ValueError("图像数据为空，请重新选择本地图像。")

    header, separator, encoded = data_url.partition(",")
    if not separator or not header.lower().startswith("data:image/") or ";base64" not in header.lower():
        raise ValueError("仅支持 base64 编码的图像数据。")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图像数据解码失败，请重新上传。") from exc
    if not raw:
        raise ValueError("图像数据为空，请重新上传。")
    if len(raw) > MAX_DECODED_IMAGE_BYTES:
        raise ValueError("图像文件过大，请上传不超过 40 MiB 的图像。")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("无法识别上传的图像，请使用 JPG、PNG、WEBP 或 TIFF。") from exc
    if image.width <= 0 or image.height <= 0:
        raise ValueError("上传的图像尺寸无效。")
    return image


def png_data_url(array: np.ndarray, mode: str = "RGB") -> str:
    """Encode an array as a browser-safe PNG data URL."""

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array), mode=mode).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def release_model_resources(instance: Any | None) -> None:
    """Drop one cached predictor and release allocator-held CUDA memory when available."""

    if instance is None:
        return
    del instance
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
