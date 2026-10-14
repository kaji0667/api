"""Offline full-object CLIP matching; no production dispatch or new candidates.

Square mean-color padding preserves both ends of a rectangular proposal before
the unchanged CLIP transform. Unknown role grammar uses the complete source;
it does not invent a target phrase or discard negation/relations/attributes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .crop_identity import FrozenCLIPCropScorer, _official_transform
from .reference_plan import parse_reference_plan


def square_pad(image: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    side = max(image.size)
    result = Image.new("RGB", (side, side), fill)
    result.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return result


def whole_transform(configuration: dict[str, Any]):
    base = _official_transform(configuration)
    fill = tuple(round(255 * value) for value in configuration["mean"])
    return lambda image: base(square_pad(image, fill))


class WholeBBoxCLIPScorer(FrozenCLIPCropScorer):
    def __init__(self, clip_model, encode_descriptions, *, batch_size: int = 8):
        super().__init__(clip_model, encode_descriptions, batch_size=batch_size,
                         transform_factory=whole_transform)


def select_whole_crop(records: list[dict], baseline_scores, baseline: int | None) -> int | None:
    """Original native scores + the existing fixed .15 identity correction."""
    if not records or baseline is None:
        return baseline
    scores = np.asarray(baseline_scores, dtype=np.float64)
    if (scores.shape != (len(records),) or not np.isfinite(scores).all()
            or type(baseline) is not int or not 0 <= baseline < len(records)
            or any(row.get("index") != i or row.get("status") != "succeeded"
                   for i, row in enumerate(records))):
        return baseline
    try:
        contrast = np.asarray([row["contrast_score"] for row in records], dtype=np.float64)
    except (ValueError, TypeError, KeyError):
        return baseline
    if contrast.shape != scores.shape or not np.isfinite(contrast).all():
        return baseline
    span = float(np.ptp(contrast))
    correction = (contrast - contrast.min()) / span if span > 1e-12 else np.zeros_like(contrast)
    final = scores + .15 * correction
    tied = np.flatnonzero(final.max() - final <= 1e-12).tolist()
    return baseline if baseline in tied else tied[0]


def rank_whole_crop(scorer, image, description: str, boxes, baseline_scores,
                    native_index: int | None, guarded_index: int | None) -> dict:
    plan = parse_reference_plan(description)
    target = plan.target_description if plan else description
    references = [ref.description for ref in plan.references] if plan else []
    result = {"parsed": plan is not None, "target": target, "references": references,
              "records": [], "status": "empty_candidates", "crop_count": 0,
              "pad_guarded_index": guarded_index, "pad_full_index": guarded_index}
    if not len(boxes):
        return result
    try:
        evidence = scorer.score(image, boxes, target, references)
        records = evidence["records"]
        if (len(records) != len(boxes)
                or any(row.get("index") != index or row.get("status") != "succeeded"
                       or not isinstance(row.get("contrast_score"), (int, float))
                       or not np.isfinite(row["contrast_score"])
                       for index, row in enumerate(records))):
            result["status"] = "incomplete_matching_evidence"
            return result
        selected = select_whole_crop(records, baseline_scores, native_index)
        result.update(records=records, crop_count=evidence["encoded_crop_count"],
                      status="succeeded", pad_full_index=selected)
        if plan is not None:
            result["pad_guarded_index"] = selected
    except Exception:
        result["status"] = "optional_matching_failed"
    return result
