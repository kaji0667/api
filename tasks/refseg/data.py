"""Dataset, text-cache, and sampling helpers for referring segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import WeightedRandomSampler

from dataset.rrsisd_refseg_dataset import RRSISDRefSegDataset, decode_compressed_rle_counts

from .runtime import class_prompts, dataset_type, resolve_data_root


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(value)).strip("-") or "default"


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def build_rrsisd_text_embedding_cache(
    args: argparse.Namespace,
    data: Dict[str, Any],
    split_paths: Dict[str, Path],
    device: torch.device,
) -> Dict[str, Path]:
    """Build or reuse per-sample OpenCLIP text embeddings for RRSIS-D splits."""
    if dataset_type(data) != "rrsisd_refseg":
        return {}
    if args.text_encoder != "openclip":
        raise ValueError("RRSIS-D free-text segmentation requires --text-encoder openclip.")
    if int(args.text_embed_batch_size) <= 0:
        raise ValueError("--text-embed-batch-size must be > 0")

    embedding_root = (
        resolve_data_root(data)
        / f"openclip_{_safe_tag(args.text_model_name)}_{_safe_tag(args.text_pretrained)}_tokens"
    )
    paths = {split: embedding_root / f"{split}_text_embeddings.pt" for split in split_paths}
    missing = [split for split, path in paths.items() if args.text_overwrite or not path.exists()]
    if not missing:
        print(f"RRSIS-D text embeddings: using cache {embedding_root}")
        return paths

    from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder

    encoder = OpenCLIPTextEncoder(
        model_name=args.text_model_name,
        pretrained=args.text_pretrained,
        device=str(device),
        precision=args.text_precision,
        normalize=True,
    )
    if int(encoder.embedding_dim) != 768:
        raise ValueError(
            f"OpenCLIP text dim is {int(encoder.embedding_dim)}, but yolov12-semseg.yaml expects 768. "
            "Use ViT-L-14 or update TextPromptSegment text_dim."
        )

    embedding_root.mkdir(parents=True, exist_ok=True)
    for split in missing:
        rows = _load_jsonl_rows(split_paths[split])
        ids = [str(row["id"]) for row in rows]
        texts = [str(row.get("text", "")).strip() for row in rows]
        print(f"Building RRSIS-D {split} text embeddings: {len(texts)} samples -> {paths[split]}")
        embeddings, token_masks = encoder.encode(
            texts,
            batch_size=int(args.text_embed_batch_size),
            return_tokens=True,
        )
        payload = {
            "ids": ids,
            "texts": texts,
            "embeddings": embeddings.float().cpu(),
            "token_masks": token_masks.bool().cpu(),
            "metadata": {
                "dataset_type": "rrsisd_refseg",
                "split": split,
                "model_name": args.text_model_name,
                "pretrained": args.text_pretrained,
                "embedding_dim": int(embeddings.shape[-1]) if embeddings.ndim >= 2 else 0,
                "embedding_kind": "token_features",
            },
        }
        torch.save(payload, paths[split])
    return paths


def build_prompt_embeddings(
    args: argparse.Namespace,
    data: Dict[str, Any],
    names: List[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if dataset_type(data) == "rrsisd_refseg":
        return None
    if not args.text_queries or args.text_encoder != "openclip":
        return None

    from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder

    prompts = class_prompts(data, names)
    encoder = OpenCLIPTextEncoder(
        model_name=args.text_model_name,
        pretrained=args.text_pretrained,
        device=str(device),
        precision=args.text_precision,
        normalize=True,
    )
    embeddings = encoder.encode(prompts, batch_size=len(prompts), return_tokens=False)
    if embeddings.ndim != 2:
        raise RuntimeError(f"OpenCLIP prompt embeddings must be [N, D], got {tuple(embeddings.shape)}")
    if int(embeddings.shape[1]) != 768:
        raise ValueError(
            f"OpenCLIP text dim is {int(embeddings.shape[1])}, but yolov12-semseg.yaml currently expects 768. "
            "Use ViT-L-14 or update the TextPromptSegment text_dim argument in the model yaml."
        )
    print(
        f"OpenCLIP prompt embeddings: model={args.text_model_name}, "
        f"pretrained={args.text_pretrained}, shape={tuple(embeddings.shape)}"
    )
    return embeddings.to(device=device, dtype=torch.float32)


def build_semseg_dataset(
    data: Dict[str, Any],
    split_path: Path,
    args: argparse.Namespace,
    split: str,
    text_embedding_file: Optional[Path] = None,
):
    dtype = dataset_type(data)
    if dtype == "rrsisd_refseg":
        return RRSISDRefSegDataset(
            split_path,
            image_size=args.imgsz,
            normalize=True,
            text_embedding_file=text_embedding_file,
            augment=bool(args.augment and split == "train"),
            hflip_prob=args.augment_hflip,
            vflip_prob=args.augment_vflip,
            color_jitter=args.augment_color_jitter,
            directional_flip_policy=args.augment_direction_policy,
            empty_mask_policy=args.empty_mask_policy,
        )
    raise ValueError(f"Unsupported semantic dataset_type: {dtype}")


def _rle_foreground_area(rle: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2 or counts is None:
        return None
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        return None
    decoded_counts = decode_compressed_rle_counts(counts) if isinstance(counts, str) else list(counts)
    foreground = sum(float(count) for index, count in enumerate(decoded_counts) if index % 2 == 1)
    return foreground, float(height * width)


def _segmentation_area_ratio(segmentation: Any) -> Optional[float]:
    items = segmentation if isinstance(segmentation, list) else [segmentation]
    foreground = 0.0
    total = None
    for item in items:
        if not isinstance(item, dict):
            continue
        area = _rle_foreground_area(item)
        if area is None:
            continue
        item_foreground, item_total = area
        foreground += item_foreground
        total = item_total if total is None else total
    if total is None or total <= 0:
        return None
    return max(foreground, 0.0) / total


def _sample_area_ratio(row: Dict[str, Any]) -> Optional[float]:
    mask_area = _segmentation_area_ratio(row.get("segmentation"))
    if mask_area is not None:
        return mask_area

    bbox = row.get("bbox")
    width = float(row.get("width") or 0)
    height = float(row.get("height") or 0)
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or width <= 0 or height <= 0:
        return None
    return max(float(bbox[2]), 0.0) * max(float(bbox[3]), 0.0) / max(width * height, 1.0)


def build_text_query_sampler(
    dataset,
    small_target_boost: float = 1.0,
    small_target_area: float = 0.0025,
    generator: Optional[torch.Generator] = None,
) -> Optional[WeightedRandomSampler]:
    query_items = getattr(dataset, "query_items", None)
    rows = getattr(dataset, "rows", None)

    if query_items:
        class_indices = [int(class_idx) for _, class_idx in query_items]
        area_ratios = [None] * len(class_indices)
    elif rows:
        class_indices = [int(row.get("class_idx", 0)) for row in rows]
        area_ratios = [_sample_area_ratio(row) for row in rows]
    else:
        return None

    counts: Dict[int, int] = {}
    for class_idx in class_indices:
        counts[class_idx] = counts.get(class_idx, 0) + 1

    boost = max(float(small_target_boost), 1.0)
    threshold = max(float(small_target_area), 0.0)
    weights_list = []
    boosted = 0
    for class_idx, area_ratio in zip(class_indices, area_ratios):
        weight = 1.0 / counts[class_idx]
        if boost > 1.0 and area_ratio is not None and area_ratio <= threshold:
            weight *= boost
            boosted += 1
        weights_list.append(weight)

    weights = torch.as_tensor(weights_list, dtype=torch.double)
    print(f"text query class counts: {dict(sorted(counts.items()))}")
    if boosted:
        print(
            f"small target sampler boost: {boosted}/{len(weights_list)} samples x{boost:g} "
            f"(mask area <= {threshold:g})"
        )
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


__all__ = (
    "build_prompt_embeddings",
    "build_rrsisd_text_embedding_cache",
    "build_semseg_dataset",
    "build_text_query_sampler",
)
