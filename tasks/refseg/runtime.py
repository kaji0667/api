"""Shared runtime helpers for referring-segmentation training and evaluation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ultralytics.nn import SemanticSegmentationModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_existing_path(path: str | Path) -> Optional[Path]:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate

    raw = str(path)
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        wsl_candidate = Path(f"/mnt/{drive}/{rest}")
        if wsl_candidate.exists():
            return wsl_candidate
    return None


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint from the caller's CWD, then from the HFSA-main directory."""
    resolved = resolve_existing_path(path)
    if resolved is not None:
        return resolved.resolve()

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        project_candidate = PROJECT_ROOT / candidate
        if project_candidate.exists():
            return project_candidate.resolve()

    raise FileNotFoundError(f"Checkpoint not found: {path}")


def resolve_data_root(data: Dict[str, Any]) -> Path:
    configured = resolve_existing_path(data["path"])
    if configured is not None:
        return configured

    yaml_dir = data.get("_yaml_dir")
    if yaml_dir:
        yaml_root = Path(yaml_dir).expanduser()
        if yaml_root.exists():
            return yaml_root

    return Path(data["path"]).expanduser()


def resolve_split_path(data: Dict[str, Any], split: str) -> Path:
    root = resolve_data_root(data)
    split_path = Path(data[split]).expanduser()
    return split_path if split_path.is_absolute() else root / split_path


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def load_pretrained_backbone(model: SemanticSegmentationModel, weights: str, device: torch.device) -> None:
    """Load matching pretrained YOLO weights while leaving the semantic segmentation head random."""
    if not weights:
        print("pretrained weights: disabled")
        return

    weights_path = Path(weights).expanduser()
    if not weights_path.exists() and not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)
    source = ckpt.get("model", ckpt.get("ema", ckpt)) if isinstance(ckpt, dict) else ckpt
    if isinstance(source, torch.nn.Module):
        source_state = source.float().state_dict()
    elif isinstance(source, dict):
        source_state = source
    else:
        raise TypeError(f"Unsupported pretrained weights format: {type(source)}")

    source_state = {k.removeprefix("module."): v for k, v in source_state.items() if isinstance(v, torch.Tensor)}
    target_state = model.state_dict()
    head_prefix = f"model.{len(model.model) - 1}."
    matched = {
        k: v
        for k, v in source_state.items()
        if k in target_state and target_state[k].shape == v.shape and not k.startswith(head_prefix)
    }
    model.load_state_dict(matched, strict=False)
    print(
        f"pretrained weights: loaded {len(matched)}/{len(target_state)} tensors from {weights_path}; "
        f"skipped head prefix {head_prefix}"
    )


def configure_trainable_layers(model: SemanticSegmentationModel, args: argparse.Namespace) -> int:
    layers = list(getattr(model, "model", []))
    if not layers:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.train_head_only:
        freeze_until = len(layers) - 1
    elif args.freeze_backbone and args.freeze_neck:
        freeze_until = len(layers) - 1
    elif args.freeze_backbone:
        freeze_until = min(9, len(layers) - 1)
    elif args.freeze_neck:
        freeze_until = 0
    else:
        freeze_until = 0

    if args.freeze_neck and not args.train_head_only and not args.freeze_backbone:
        freeze_indices = set(range(9, max(len(layers) - 1, 9)))
    else:
        freeze_indices = set(range(freeze_until))

    for i, layer in enumerate(layers):
        freeze = i in freeze_indices
        for param in layer.parameters():
            param.requires_grad = not freeze

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {trainable:,}; frozen params: {frozen:,}")
    return trainable


def predict_with_optional_text(
    model: SemanticSegmentationModel,
    batch: Dict[str, Any],
    text_queries: bool,
    prompt_embeddings: Optional[torch.Tensor] = None,
):
    """Run the model with class-conditioned prompt embeddings when text queries are enabled."""
    if text_queries:
        class_idx = batch.get("class_idx")
        text_embedding = batch.get("text_embedding")
        if text_embedding is None and prompt_embeddings is not None and class_idx is not None:
            text_embedding = prompt_embeddings[class_idx]
        return model(
            batch["img"],
            class_idx=class_idx,
            text_embedding=text_embedding,
            text_token_mask=batch.get("text_token_mask"),
        )
    return model(batch["img"])


def class_names(data: Dict[str, Any], nc: int) -> List[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names.get(i, names.get(str(i), i))) for i in range(nc)]
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    return [str(i) for i in range(nc)]


def class_prompts(data: Dict[str, Any], names: List[str]) -> List[str]:
    prompts = data.get("prompts", {})
    if not isinstance(prompts, dict):
        prompts = {}
    return [str(prompts.get(name, name)) for name in names]


def dataset_type(data: Dict[str, Any]) -> str:
    return str(data.get("dataset_type", "rrsisd_refseg") or "rrsisd_refseg").strip().lower()


def parse_val_thresholds(raw: str) -> List[float]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not 0.0 < value < 1.0:
            raise ValueError(f"--val-thresholds values must be between 0 and 1, got {value}")
        values.append(value)
    return values or [0.5]


def validation_selection_score(metrics: Dict[str, Any], select_metric: str) -> float:
    if select_metric in {"iou", "oiou"}:
        return float(metrics["oiou"])
    if select_metric == "miou":
        return float(metrics["official_miou"])
    if select_metric == "f1":
        return float(metrics["f1"])
    if select_metric == "fbeta":
        return float(metrics["fbeta"])
    if select_metric == "precision":
        return float(metrics["precision"])
    raise ValueError(f"Unsupported validation selection metric: {select_metric}")


__all__ = (
    "PROJECT_ROOT",
    "class_names",
    "class_prompts",
    "configure_trainable_layers",
    "dataset_type",
    "load_pretrained_backbone",
    "move_batch_to_device",
    "parse_val_thresholds",
    "predict_with_optional_text",
    "resolve_checkpoint_path",
    "resolve_data_root",
    "resolve_existing_path",
    "resolve_split_path",
    "seed_worker",
    "set_random_seed",
    "validation_selection_score",
)
