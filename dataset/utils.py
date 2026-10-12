from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from dataset.voc_object_dataset import (
    ObjectLevelDescriptionDataset,
    ImageLevelDescriptionDataset,
    load_object_descriptions,
    load_sample_rows,
    prepare_voc_image_level_dataset,
    prepare_voc_object_level_dataset,
    validate_prepared_yolo_dataset,
)


def _required_embedding_splits() -> List[str]:
    # Train/val are required for text-guided training.
    return ["train", "val"]


def _safe_tag(value: str) -> str:
    return str(value or "").lower().replace("/", "-").replace(" ", "")


def _resolve_model_prepared_root(args: Any) -> Path:
    base_root = Path(str(getattr(args, "prepared_dir", "") or "")).expanduser().resolve()
    if not str(base_root):
        raise ValueError("Missing --prepared-dir/--prepared_dir")
    model_name = str(getattr(args, "text_model_name", "ViT-L-14") or "ViT-L-14")
    pretrained = str(getattr(args, "text_pretrained", "openai") or "openai")
    model_tag = f"openclip_{_safe_tag(model_name)}_{_safe_tag(pretrained)}"
    suffix = f"_{model_tag}"
    if base_root.name.endswith(suffix):
        return base_root
    return base_root.with_name(f"{base_root.name}{suffix}")


def _has_required_embeddings(prepared_root: Path, required_splits: List[str]) -> bool:
    for split in required_splits:
        emb_file = prepared_root / f"{split}_text_embeddings.pt"
        if not emb_file.exists():
            return False
    return True


def _resolve_data_yaml_from_args(args: Any) -> str:
    raw_data = str(getattr(args, "data", "") or "").strip()
    if not raw_data:
        return ""
    data_path = Path(raw_data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"--data points to a missing file: {data_path}")
    return str(data_path)


def build_object_level_dataset(args: Any, device: str = "auto") -> Dict[str, Any]:
    """Public API: build object-level records from VOC XML and text embeddings."""
    return prepare_voc_object_level_dataset(args=args, device=device)


def prepare_dataset(args: Any, device: str = "auto") -> str:
    """
    Prepare dataset for text-guided training and return data.yaml path.

    Priority:
    1) If --data is provided, use it directly.
    2) Reuse an existing valid prepared dataset with required embedding files.
    3) Otherwise rebuild image-level dataset and embeddings.
    """
    direct_yaml = _resolve_data_yaml_from_args(args)
    if direct_yaml:
        return direct_yaml

    prepared_root = _resolve_model_prepared_root(args)
    # Keep args in sync so downstream builders/writers use the model-suffixed outer directory.
    setattr(args, "prepared_dir", str(prepared_root))
    required_splits = _required_embedding_splits()

    if prepared_root.exists():
        validation = validate_prepared_yolo_dataset(str(prepared_root), required_splits=["train", "val"])
        data_yaml = str((prepared_root / "data.yaml").resolve())
        if validation.get("valid", False):
            if _has_required_embeddings(prepared_root, required_splits):
                return data_yaml

            from dataset.precompute_text_embeddings import build_openclip_text_embeddings

            build_openclip_text_embeddings(
                prepared_dir=str(prepared_root),
                splits=required_splits,
                output_dir=str(prepared_root),
                model_name=str(getattr(args, "text_model_name", "ViT-L-14") or "ViT-L-14"),
                pretrained=str(getattr(args, "text_pretrained", "openai") or "openai"),
                device=device,
                precision=str(getattr(args, "text_precision", "auto") or "auto"),
                batch_size=128,
                overwrite=bool(getattr(args, "text_overwrite", False)),
            )
            if _has_required_embeddings(prepared_root, required_splits):
                return data_yaml

    result = prepare_voc_image_level_dataset(args=args, device=device)
    data_yaml = str(result.get("data_yaml", "") or "").strip()
    if not data_yaml:
        data_yaml = str((prepared_root / "data.yaml").resolve())

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Dataset preparation failed, missing data.yaml: {data_yaml}")

    return data_yaml


__all__ = [
    "prepare_dataset",
    "build_object_level_dataset",
    "prepare_voc_object_level_dataset",
    "prepare_voc_image_level_dataset",
    "validate_prepared_yolo_dataset",
    "load_object_descriptions",
    "load_sample_rows",
    "ObjectLevelDescriptionDataset",
    "ImageLevelDescriptionDataset",
]
