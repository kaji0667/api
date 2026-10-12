from __future__ import annotations

import argparse
import json
import pickle
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_DIRECTIONAL_TEXT_RE = re.compile(
    r"\b("
    r"left|right|top|bottom|upper|lower|above|below|middle|center|centre|"
    r"north|south|east|west|northeast|northwest|southeast|southwest|"
    r"leftmost|rightmost|topmost|bottommost"
    r")\b",
    re.IGNORECASE,
)

_HORIZONTAL_DIRECTION_RE = re.compile(
    r"\b(left|right|east|west|northeast|northwest|southeast|southwest|leftmost|rightmost)\b",
    re.IGNORECASE,
)

_VERTICAL_DIRECTION_RE = re.compile(
    r"\b(top|bottom|upper|lower|above|below|north|south|northeast|northwest|southeast|southwest|topmost|bottommost)\b",
    re.IGNORECASE,
)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _yaml_scalar(value: Any) -> str:
    text = str(value)
    if not text or any(ch in text for ch in ":#[]{}&,*!|>'\"%@`"):
        return json.dumps(text, ensure_ascii=False)
    return text


def _write_data_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"dataset_type: {data['dataset_type']}",
        f"path: {_yaml_scalar(data['path'])}",
        f"train: {data['train']}",
        f"val: {data['val']}",
        f"test: {data['test']}",
        f"nc: {data['nc']}",
        "names:",
    ]
    lines.extend(f"  {i}: {_yaml_scalar(name)}" for i, name in enumerate(data["names"]))
    lines.extend(
        [
            f"ignore_index: {data['ignore_index']}",
            "mask_value_mapping: rrsisd_refseg_binary_0_1",
            f"source_root: {_yaml_scalar(data['source_root'])}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _has_directional_text(text: str) -> bool:
    return bool(_DIRECTIONAL_TEXT_RE.search(str(text)))


def _directional_axes(text: str) -> Tuple[bool, bool]:
    """Return whether text constrains the horizontal and vertical axes."""
    value = str(text)
    return bool(_HORIZONTAL_DIRECTION_RE.search(value)), bool(_VERTICAL_DIRECTION_RE.search(value))


def _load_refs(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        refs = pickle.load(f)
    if not isinstance(refs, list):
        raise TypeError(f"RRSIS-D refs must be a list, got {type(refs)}")
    return refs


def _first_sentence(ref: Dict[str, Any]) -> str:
    sentences = ref.get("sentences", [])
    if not sentences:
        return ""
    first = sentences[0]
    if isinstance(first, dict):
        return str(first.get("sent") or first.get("raw") or "").strip()
    return str(first).strip()


def _category_names(categories: Sequence[Dict[str, Any]]) -> List[str]:
    if not categories:
        raise ValueError("RRSIS-D instances.json has no categories.")
    max_id = max(int(cat["id"]) for cat in categories)
    names = [str(i) for i in range(max_id + 1)]
    for cat in categories:
        names[int(cat["id"])] = str(cat["name"])
    return names


def collect_rrsisd_rows(root: str | Path, split: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    root = Path(root).expanduser().resolve()
    split = str(split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported RRSIS-D split: {split}")

    refs_path = root / "rrsisd" / "refs(unc).p"
    instances_path = root / "rrsisd" / "instances.json"
    image_dir = root / "images" / "rrsisd" / "JPEGImages"
    if not refs_path.exists():
        raise FileNotFoundError(f"Missing RRSIS-D refs file: {refs_path}")
    if not instances_path.exists():
        raise FileNotFoundError(f"Missing RRSIS-D instances file: {instances_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing RRSIS-D image directory: {image_dir}")

    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    names = _category_names(instances.get("categories", []))
    annotations = instances.get("annotations", [])
    ann_by_id = {int(ann["id"]): ann for ann in annotations}
    images = instances.get("images", [])
    image_by_id = {int(img["id"]): img for img in images}

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for ref in _load_refs(refs_path):
        if str(ref.get("split", "")).lower() != split:
            continue
        ann_id = int(ref["ann_id"])
        image_id = int(ref["image_id"])
        ann = ann_by_id.get(ann_id)
        image_info = image_by_id.get(image_id, {})
        if ann is None:
            missing.append(f"ann_id={ann_id}")
            continue

        file_name = str(ref.get("file_name") or image_info.get("file_name") or f"{image_id:05d}.jpg")
        image_path = image_dir / file_name
        if not image_path.exists():
            missing.append(str(image_path))
            continue

        category_id = ref.get("category_id")
        if category_id is None:
            category_id = ann.get("category_id", ann.get("categories_id"))
        if category_id is None:
            raise ValueError(f"Missing category_id for RRSIS-D ann_id={ann_id}, ref_id={ref.get('ref_id', ann_id)}")
        class_idx = int(category_id)
        if not 0 <= class_idx < len(names):
            raise ValueError(
                f"Invalid category_id={class_idx} for RRSIS-D ann_id={ann_id}; expected 0 <= id < {len(names)}"
            )
        text = _first_sentence(ref)
        if not text:
            raise ValueError(f"Missing referring expression for RRSIS-D ann_id={ann_id}, ref_id={ref.get('ref_id', ann_id)}")

        rows.append(
            {
                "id": f"{split}_{int(ref.get('ref_id', ann_id))}",
                "split": split,
                "image": str(image_path),
                "file_name": file_name,
                "image_id": image_id,
                "ann_id": ann_id,
                "ref_id": int(ref.get("ref_id", ann_id)),
                "text": text,
                "category_id": class_idx,
                "class_idx": class_idx,
                "category_name": names[class_idx] if 0 <= class_idx < len(names) else str(class_idx),
                "class_name": names[class_idx] if 0 <= class_idx < len(names) else str(class_idx),
                "height": int(image_info.get("height", 0) or ann.get("height", 0) or 0),
                "width": int(image_info.get("width", 0) or ann.get("width", 0) or 0),
                "bbox": ann.get("bbox", []),
                "segmentation": ann.get("segmentation", []),
                "mask_value_mapping": "RLE foreground -> 1; background -> 0",
            }
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} RRSIS-D item(s) for split {split}. First entries: {preview}")
    rows, _ = clean_rrsisd_rows(rows, empty_mask_policy="drop", context=f"RRSIS-D {split} preparation")
    return rows, names


def prepare_rrsisd_refseg_dataset(
    root: str | Path = "data/RRSIS-D",
    output_dir: str | Path = "pre_datasets/RRSIS-D_refseg",
) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    names: List[str] = []
    for split in ("train", "val", "test"):
        rows, split_names = collect_rrsisd_rows(root, split)
        rows_by_split[split] = rows
        names = split_names
        _write_jsonl(output_dir / f"{split}.jsonl", rows)

    data = {
        "dataset_type": "rrsisd_refseg",
        "path": str(output_dir),
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "nc": len(names),
        "names": names,
        "ignore_index": 255,
        "source_root": str(root),
    }
    _write_data_yaml(output_dir / "data.yaml", data)

    return {
        "root": str(root),
        "output_dir": str(output_dir),
        "data_yaml": str(output_dir / "data.yaml"),
        "train": len(rows_by_split["train"]),
        "val": len(rows_by_split["val"]),
        "test": len(rows_by_split["test"]),
        "classes": names,
    }


def decode_compressed_rle_counts(counts: str | Sequence[int]) -> List[int]:
    if isinstance(counts, (list, tuple)):
        return [int(x) for x in counts]
    if not isinstance(counts, str):
        raise TypeError(f"Unsupported RLE counts type: {type(counts)}")

    decoded: List[int] = []
    p = 0
    while p < len(counts):
        value = 0
        shift = 0
        while True:
            c = ord(counts[p]) - 48
            p += 1
            value |= (c & 0x1F) << shift
            shift += 5
            if not (c & 0x20):
                if c & 0x10:
                    value |= -1 << shift
                break
        if len(decoded) > 2:
            value += decoded[-2]
        decoded.append(int(value))
    return decoded


def _validated_rle_counts(rle: Dict[str, Any]) -> Tuple[int, int, List[int]]:
    size = rle.get("size", None)
    counts = rle.get("counts", None)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"RLE size must be [height, width], got {size}")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"RLE size values must be positive, got {size}")
    decoded_counts = decode_compressed_rle_counts(counts)
    expected = height * width
    if any(int(count) < 0 for count in decoded_counts):
        raise ValueError("RLE counts must be non-negative.")
    actual = sum(int(count) for count in decoded_counts)
    if actual != expected:
        raise ValueError(f"RLE count sum {actual} does not match mask size {expected}.")
    return height, width, decoded_counts


def decode_rle_mask(rle: Dict[str, Any]):
    import numpy as np

    height, width, decoded_counts = _validated_rle_counts(rle)
    expected = height * width
    flat = np.zeros(expected, dtype=np.uint8)
    index = 0
    value = 0
    for count in decoded_counts:
        count = int(count)
        end = index + count
        if value == 1 and end > index:
            flat[index:end] = 1
        index += count
        value = 1 - value
    return flat.reshape((height, width), order="F")


def decode_segmentation_mask(segmentation: Any):
    import numpy as np

    if isinstance(segmentation, dict):
        return decode_rle_mask(segmentation)
    if isinstance(segmentation, list) and segmentation:
        masks = [decode_rle_mask(item) for item in segmentation if isinstance(item, dict)]
        if not masks:
            raise ValueError("RRSIS-D polygon segmentations are not supported in this loader.")
        merged = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            merged |= mask.astype(np.uint8)
        return merged
    raise ValueError("Missing or unsupported RRSIS-D segmentation.")


def segmentation_foreground_area(segmentation: Any) -> int:
    """Return exact encoded foreground area without allocating a dense mask."""
    items = [segmentation] if isinstance(segmentation, dict) else segmentation
    if not isinstance(items, list) or not items:
        raise ValueError("Missing or unsupported RRSIS-D segmentation.")
    rles = [item for item in items if isinstance(item, dict)]
    if not rles:
        raise ValueError("RRSIS-D polygon segmentations are not supported in this loader.")
    foreground = 0
    for rle in rles:
        _, _, decoded_counts = _validated_rle_counts(rle)
        foreground += sum(int(count) for index, count in enumerate(decoded_counts) if index % 2 == 1)
    return foreground


def clean_rrsisd_rows(
    rows: Sequence[Dict[str, Any]],
    empty_mask_policy: str = "drop",
    context: str = "RRSIS-D dataset",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate query metadata and explicitly drop, reject, or retain empty-mask annotations."""
    policy = str(empty_mask_policy).strip().lower()
    if policy not in {"drop", "error", "keep"}:
        raise ValueError(f"empty_mask_policy must be 'drop', 'error', or 'keep', got {empty_mask_policy!r}")

    cleaned: List[Dict[str, Any]] = []
    dropped_ids: List[str] = []
    seen_ids = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"Missing sample id in {context} at row {index}")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id {sample_id!r} in {context}")
        seen_ids.add(sample_id)

        text = row.get("text")
        if text is None or not str(text).strip():
            raise ValueError(f"Missing referring expression for sample_id={sample_id} in {context}")
        try:
            class_idx = int(row["class_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class_idx for sample_id={sample_id} in {context}") from exc
        if class_idx < 0:
            raise ValueError(f"class_idx must be non-negative for sample_id={sample_id}, got {class_idx}")

        foreground_area = segmentation_foreground_area(row.get("segmentation"))
        if foreground_area == 0:
            if policy == "error":
                raise ValueError(f"Empty segmentation mask for sample_id={sample_id} in {context}")
            if policy == "drop":
                dropped_ids.append(sample_id)
                continue
        cleaned.append(row)

    if dropped_ids:
        preview = ", ".join(dropped_ids[:5])
        warnings.warn(
            f"Dropped {len(dropped_ids)} empty-mask annotation(s) from {context}: {preview}",
            UserWarning,
            stacklevel=2,
        )
    return cleaned, dropped_ids


def _resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    raw = str(path)
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        wsl_candidate = Path(f"/mnt/{drive}/{rest}")
        if wsl_candidate.exists():
            return wsl_candidate

    raise FileNotFoundError(f"Path does not exist: {path}")


def _load_text_embeddings(path: str | Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Text embedding payload must be a dict, got {type(payload)}")
    ids = payload.get("ids", [])
    embeddings = payload.get("embeddings", None)
    if not isinstance(ids, (list, tuple)) or embeddings is None:
        raise ValueError(f"Text embedding payload must contain ids and embeddings: {path}")
    if int(len(ids)) != int(embeddings.shape[0]):
        raise ValueError(f"Text embedding count mismatch in {path}: {len(ids)} ids vs {embeddings.shape[0]} vectors")
    optional_masks = {
        "text_token_mask": payload.get("token_masks"),
    }
    for name, value in optional_masks.items():
        if value is not None and int(value.shape[0]) != int(len(ids)):
            raise ValueError(f"{name} count mismatch in {path}: {len(ids)} ids vs {value.shape[0]} masks")

    result: Dict[str, Any] = {}
    for i, sample_id in enumerate(ids):
        item = {"text_embedding": embeddings[i].detach().cpu().float()}
        for name, value in optional_masks.items():
            if value is not None:
                item[name] = value[i].detach().cpu().bool()
        result[str(sample_id)] = item
    return result


class RRSISDRefSegDataset:
    """RRSIS-D referring segmentation dataset: image + expression -> binary mask."""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]] | str | Path,
        image_size: Optional[int] = None,
        normalize: bool = True,
        text_embedding_file: Optional[str | Path] = None,
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        color_jitter: float = 0.15,
        directional_flip_policy: str = "axis-aware",
        empty_mask_policy: str = "drop",
    ) -> None:
        loaded_rows = _load_jsonl_rows(rows) if isinstance(rows, (str, Path)) else list(rows)
        context = str(rows) if isinstance(rows, (str, Path)) else "in-memory RRSIS-D rows"
        self.rows, self.dropped_empty_sample_ids = clean_rrsisd_rows(
            loaded_rows,
            empty_mask_policy=empty_mask_policy,
            context=context,
        )
        self.image_size = int(image_size) if image_size else None
        if self.image_size is not None and self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.normalize = bool(normalize)
        self.text_embeddings = _load_text_embeddings(text_embedding_file) if text_embedding_file else {}
        self.require_text_embedding = bool(text_embedding_file)
        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.color_jitter = float(color_jitter)
        for name, value in (("hflip_prob", self.hflip_prob), ("vflip_prob", self.vflip_prob)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if not 0.0 <= self.color_jitter <= 1.0:
            raise ValueError(f"color_jitter must be within [0, 1], got {self.color_jitter}")
        self.empty_mask_policy = str(empty_mask_policy).strip().lower()
        self.directional_flip_policy = str(directional_flip_policy).strip().lower()
        if self.directional_flip_policy not in {"legacy", "axis-aware"}:
            raise ValueError(
                "directional_flip_policy must be 'legacy' or 'axis-aware', "
                f"got {directional_flip_policy!r}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _read_image(path: str):
        try:
            import cv2

            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except ImportError:
            import numpy as np
            from PIL import Image

            with Image.open(path) as im:
                return np.asarray(im.convert("RGB"))

    @staticmethod
    def _resize(image, mask, image_size: int):
        try:
            import cv2

            size = (image_size, image_size)
            return (
                cv2.resize(image, size, interpolation=cv2.INTER_LINEAR),
                cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST),
            )
        except ImportError:
            import numpy as np
            from PIL import Image

            size = (image_size, image_size)
            image_resized = Image.fromarray(image).resize(size, Image.BILINEAR)
            mask_resized = Image.fromarray(mask).resize(size, Image.NEAREST)
            return np.asarray(image_resized), np.asarray(mask_resized)

    @staticmethod
    def _align_mask_to_image(image, mask):
        """Resize an annotation mask to the decoded image coordinate size when metadata differs."""
        image_height, image_width = int(image.shape[0]), int(image.shape[1])
        if tuple(mask.shape[:2]) == (image_height, image_width):
            return mask
        try:
            import cv2

            return cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        except ImportError:
            import numpy as np
            from PIL import Image

            resized = Image.fromarray(mask).resize((image_width, image_height), Image.NEAREST)
            return np.asarray(resized)

    def _augment(self, image, mask, text: str = ""):
        import numpy as np

        if self.directional_flip_policy == "legacy":
            has_direction = _has_directional_text(text)
            allow_hflip = not has_direction
            allow_vflip = not has_direction
        else:
            has_horizontal_direction, has_vertical_direction = _directional_axes(text)
            allow_hflip = not has_horizontal_direction
            allow_vflip = not has_vertical_direction

        # Always consume one draw per axis so policy ablations keep all later RNG draws aligned.
        hflip_draw = np.random.random()
        vflip_draw = np.random.random()
        if allow_hflip and self.hflip_prob > 0 and hflip_draw < self.hflip_prob:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if allow_vflip and self.vflip_prob > 0 and vflip_draw < self.vflip_prob:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        if self.color_jitter > 0:
            image_f = image.astype(np.float32)
            brightness = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            contrast = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            mean = image_f.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image_f - mean) * contrast + mean, 0.0, 255.0)
            image = np.clip(image * brightness, 0.0, 255.0).astype(np.uint8)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.rows[index]
        image_path = _resolve_existing_path(row["image"])
        image = self._read_image(str(image_path))
        mask = decode_segmentation_mask(row["segmentation"])
        mask = self._align_mask_to_image(image, mask)

        if self.image_size:
            image, mask = self._resize(image, mask, self.image_size)

        text = str(row["text"])
        if self.augment:
            image, mask = self._augment(image, mask, text)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        if self.normalize:
            image_tensor = image_tensor / 255.0
        mask_tensor = torch.from_numpy(mask.astype("uint8")).contiguous().long()

        sample_id = str(row["id"])
        sample = {
            "img": image_tensor,
            "mask": mask_tensor,
            "im_file": str(image_path),
            "sample_id": sample_id,
            "text": text,
            "prompt": text,
            "class_idx": int(row["class_idx"]),
            "class_name": str(row["class_name"]),
            "category_name": str(row["category_name"]),
            "query_id": sample_id,
            "ann_id": int(row["ann_id"]),
            "ref_id": int(row["ref_id"]),
        }
        if self.text_embeddings:
            embedding_item = self.text_embeddings.get(sample_id)
            if embedding_item is None:
                raise KeyError(f"Missing text embedding for RRSIS-D sample_id={sample_id}")
            if isinstance(embedding_item, dict):
                sample.update(embedding_item)
            else:
                sample["text_embedding"] = embedding_item
        elif self.require_text_embedding:
            raise KeyError(f"Missing text embedding file entry for RRSIS-D sample_id={sample_id}")
        return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RRSIS-D referring segmentation metadata.")
    parser.add_argument("--root", type=str, default="data/RRSIS-D", help="RRSIS-D root directory.")
    parser.add_argument("--output-dir", type=str, default="pre_datasets/RRSIS-D_refseg", help="Output metadata dir.")
    return parser.parse_args()


def main() -> None:
    result = prepare_rrsisd_refseg_dataset(**vars(parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
