"""Offline identity ranking of unchanged, real full-description candidates.

This module accepts an existing frozen CLIP and never constructs a model,
generates boxes, crops a search ROI, reads annotations, or selects an API backend.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch

from .reference_plan import ReferencePlan, parse_reference_plan


SATELLITE_TEMPLATE = "a satellite remote sensing photo of {target}"
GEOMETRY_WEIGHT = .15
TIE_EPSILON = 1e-12


class CropIdentityError(ValueError):
    """Bounded failure code; raw exceptions are not prediction diagnostics."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _official_transform(configuration: dict[str, Any]) -> Callable:
    from open_clip.transform import PreprocessCfg, image_transform_v2

    return image_transform_v2(PreprocessCfg(**configuration), is_train=False)


class FrozenCLIPCropScorer:
    """Reuse one loaded, immutable CLIP for all target/reference crop scores."""

    def __init__(
        self, clip_model: Any, encode_descriptions: Callable[[Sequence[str]], torch.Tensor],
        *, batch_size: int = 8, transform_factory: Callable | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise CropIdentityError("invalid_crop_batch_size")
        if (clip_model is None or not callable(getattr(clip_model, "encode_image", None))
                or not callable(encode_descriptions) or getattr(clip_model, "training", True)):
            raise CropIdentityError("frozen_clip_unavailable")
        try:
            parameters = list(clip_model.parameters())
            if not parameters or any(parameter.requires_grad for parameter in parameters):
                raise CropIdentityError("clip_not_frozen")
            self.device = parameters[0].device
            self.dtype = parameters[0].dtype
            configuration = dict(clip_model.visual.preprocess_cfg)
            required = {"size", "mode", "mean", "std", "interpolation", "resize_mode", "fill_color"}
            if not required.issubset(configuration) or configuration["mode"] != "RGB":
                raise CropIdentityError("clip_transform_unavailable")
            self.transform = (transform_factory or _official_transform)(configuration)
            if not callable(self.transform):
                raise CropIdentityError("clip_transform_unavailable")
        except CropIdentityError:
            raise
        except Exception:
            raise CropIdentityError("clip_transform_unavailable") from None
        self.model = clip_model
        self.encode_descriptions = encode_descriptions
        self.batch_size = batch_size
        self.preprocess_configuration = configuration

    def score(self, image_bgr: np.ndarray, candidate_boxes: Sequence[Sequence[float]],
              target: str, references: Sequence[str]) -> dict[str, Any]:
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
                or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
                or min(image_bgr.shape[:2]) <= 0):
            raise CropIdentityError("invalid_crop_image")
        phrases = [target, *references]
        if any(not isinstance(text, str) or not text.strip() for text in phrases):
            raise CropIdentityError("invalid_identity_text")
        texts = [text for phrase in phrases for text in (phrase, SATELLITE_TEMPLATE.format(target=phrase))]
        try:
            text_features = self.encode_descriptions(texts)
        except Exception as failure:
            code = getattr(failure, "code", None)
            raise CropIdentityError("text_context_exceeded" if code == "text_context_exceeded"
                                    else "clip_text_encoding_failed") from None
        if (not isinstance(text_features, torch.Tensor) or text_features.ndim != 2
                or text_features.shape[0] != len(texts) or not torch.isfinite(text_features).all()):
            raise CropIdentityError("invalid_clip_text_features")
        text_features = text_features.detach().float().to(self.device)
        norms = text_features.norm(dim=-1, keepdim=True)
        if bool((norms <= 0).any()) or not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise CropIdentityError("clip_text_features_not_normalized")
        # NativeTextEncoder already normalizes once; preserve its exact vectors.
        height, width = image_bgr.shape[:2]
        try:
            boxes = np.asarray(candidate_boxes, dtype=np.float64)
            if boxes.size == 0:
                boxes = np.empty((0, 4), dtype=np.float64)
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise CropIdentityError("invalid_candidate_boxes")
        except (ValueError, TypeError, OverflowError):
            raise CropIdentityError("invalid_candidate_boxes") from None
        if not np.isfinite(boxes).all():
            raise CropIdentityError("invalid_candidate_boxes")
        records: list[dict[str, Any]] = []
        tensors: list[torch.Tensor] = []
        valid_indices: list[int] = []
        for index, source in enumerate(boxes):
            clipped = np.clip(source, 0, [width, height, width, height])
            bounds = [math.floor(clipped[0]), math.floor(clipped[1]),
                      math.ceil(clipped[2]), math.ceil(clipped[3])]
            record = {"index": index, "box_xyxy": source.tolist(), "crop_bounds_xyxy": bounds,
                      "crop_aspect_ratio": None, "status": "invalid_candidate_area",
                      "target_template_scores": None, "reference_template_scores": None,
                      "target_score": None, "reference_scores": None, "contrast_score": None}
            records.append(record)
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            x1, y1, x2, y2 = bounds
            record["crop_aspect_ratio"] = (x2 - x1) / (y2 - y1)
            rgb = np.ascontiguousarray(image_bgr[y1:y2, x1:x2, ::-1])
            try:
                tensor = self.transform(Image.fromarray(rgb, mode="RGB"))
            except Exception:
                raise CropIdentityError("clip_crop_transform_failed") from None
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 3
                    or not torch.isfinite(tensor).all()):
                raise CropIdentityError("invalid_clip_crop_tensor")
            tensors.append(tensor)
            valid_indices.append(index)
        for start in range(0, len(tensors), self.batch_size):
            try:
                batch = torch.stack(tensors[start:start + self.batch_size]).to(self.device, dtype=self.dtype)
                with torch.inference_mode():
                    features = self.model.encode_image(batch, normalize=False).float()
            except Exception:
                raise CropIdentityError("clip_image_encoding_failed") from None
            if (features.ndim != 2 or features.shape != (len(batch), text_features.shape[1])
                    or not torch.isfinite(features).all()):
                raise CropIdentityError("invalid_clip_image_features")
            norms = features.norm(dim=-1, keepdim=True)
            if bool((norms <= 0).any()):
                raise CropIdentityError("invalid_clip_image_features")
            features = features / norms.clamp_min(1e-12)
            similarities = (features @ text_features.T).detach().cpu().numpy()
            for offset, scores in enumerate(similarities):
                pairs = scores.reshape(-1, 2)
                means = pairs.mean(axis=1)
                records[valid_indices[start + offset]].update(
                    status="succeeded", target_template_scores=pairs[0].tolist(),
                    reference_template_scores=pairs[1:].tolist(), target_score=float(means[0]),
                    reference_scores=means[1:].tolist(),
                    contrast_score=float(means[0] - max(means[1:], default=0.)) if references else float(means[0]),
                )
        return {"texts": texts, "records": records, "encoded_crop_count": len(valid_indices),
                "image_encode_batch_count": math.ceil(len(valid_indices) / self.batch_size)}


def _selected(records: list[dict[str, Any]], key: str, baseline: int | None) -> int | None:
    available = [record for record in records if record["status"] == "succeeded" and record.get(key) is not None]
    if not available:
        return None
    highest = max(record[key] for record in available)
    tied = [record["index"] for record in available if highest - record[key] <= TIE_EPSILON]
    return baseline if baseline in tied else min(tied)


def _soft_geometry(box: Sequence[float], image_hw: tuple[int, int],
                   plan: ReferencePlan | None, anchors: Sequence[dict[str, Any]]) -> float:
    if plan is None:
        return 0.
    height, width = image_hw
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    scores = []
    if plan.absolute_position is not None:
        for coordinate, extent, direction in zip((cx, cy), (width, height), plan.absolute_position):
            if direction is not None:
                ratio = float(np.clip(coordinate / extent, 0, 1))
                scores.append(1 - ratio if direction == -1 else ratio if direction == 1 else 1 - 2 * abs(ratio - .5))
    for spec, anchor in zip(plan.references, anchors):
        box_ref = anchor.get("selected_box")
        if box_ref is None:
            scores.append(.5)  # Missing reference is neutral, never a hard gate.
            continue
        ax, ay = (box_ref[0] + box_ref[2]) / 2, (box_ref[1] + box_ref[3]) / 2
        displacement = {"left_of": (ax - cx) / width, "right_of": (cx - ax) / width,
                        "above": (ay - cy) / height, "below": (cy - ay) / height}.get(spec.relation)
        if displacement is not None:
            scores.append(float(np.clip(.5 + displacement, 0, 1)))
        elif spec.relation == "near":
            dx, dy = max(box_ref[0] - cx, 0., cx - box_ref[2]), max(box_ref[1] - cy, 0., cy - box_ref[3])
            scores.append(1 - float(np.clip(math.hypot(dx, dy) / (.25 * math.hypot(width, height)), 0, 1)))
        else:
            scores.append(.5)
    return float(sum(scores) / len(scores)) if scores else 0.


def rank_full_candidates(
    scorer: FrozenCLIPCropScorer, image_bgr: np.ndarray, description: str,
    candidate_boxes: Sequence[Sequence[float]], baseline_index: int | None,
    *, phrase_scores: Sequence[float] | None = None, geometry: bool = False,
    anchors: Sequence[dict[str, Any]] = (), hybrid: bool = False,
    baseline_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Parse before scoring; preserve A and the exact original candidate order."""
    if not isinstance(description, str) or not description.strip():
        raise CropIdentityError("invalid_identity_text")
    if type(geometry) is not bool or type(hybrid) is not bool:
        raise CropIdentityError("invalid_geometry_flag")
    try:
        boxes = np.asarray(candidate_boxes, dtype=np.float64)
        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise CropIdentityError("invalid_candidate_boxes")
    except (ValueError, TypeError, OverflowError):
        raise CropIdentityError("invalid_candidate_boxes") from None
    if baseline_index is not None and (type(baseline_index) is not int or not 0 <= baseline_index < len(boxes)):
        raise CropIdentityError("invalid_baseline_index")
    if phrase_scores is not None:
        values = np.asarray(phrase_scores, dtype=np.float64)
        if values.shape != (len(boxes),) or not np.isfinite(values).all():
            raise CropIdentityError("invalid_phrase_scores")
        phrase_scores = values.tolist()
    if baseline_scores is not None:
        values = np.asarray(baseline_scores, dtype=np.float64)
        if values.shape != (len(boxes),) or not np.isfinite(values).all():
            raise CropIdentityError("invalid_baseline_scores")
        baseline_scores = values.tolist()
    plan = parse_reference_plan(description)
    target = plan.target_description if plan else description
    references = [reference.description for reference in plan.references] if plan else []
    if geometry and plan is not None and len(anchors) != len(plan.references):
        raise CropIdentityError("invalid_reference_anchors")
    if geometry and plan is not None:
        for spec, anchor in zip(plan.references, anchors):
            if anchor.get("description") != spec.description or anchor.get("relation") != spec.relation:
                raise CropIdentityError("invalid_reference_anchors")
            box_ref = anchor.get("selected_box")
            if box_ref is not None and (len(box_ref) != 4 or not all(math.isfinite(float(x)) for x in box_ref)
                                        or box_ref[2] <= box_ref[0] or box_ref[3] <= box_ref[1]):
                raise CropIdentityError("invalid_reference_anchors")
    result: dict[str, Any] = {
        "original_description": description, "plan": asdict(plan) if plan else None,
        "parse_status": "parsed" if plan else "complete_description_fallback",
        "target_description": target, "reference_descriptions": references,
        "candidate_source": "full_original", "candidate_boxes": boxes.tolist(),
        "phrase_scores": phrase_scores, "baseline_index": baseline_index,
        "baseline_adjusted_scores": baseline_scores,
        "arms": {"A": {"selected_index": baseline_index,
                         "status": "succeeded" if baseline_index is not None else "no_original_selection"}},
        "score_status": "empty_candidates" if not len(boxes) else "pending", "crop_scores": [],
        "configuration": {"satellite_template": SATELLITE_TEMPLATE, "target_score": "two_cosine_arithmetic_mean",
                          "contrast_score": "target_mean_minus_max_reference_mean",
                          "geometry_weight": GEOMETRY_WEIGHT if geometry else 0., "tie_epsilon": TIE_EPSILON,
                          "tie_policy": "original_selection_then_lowest_candidate_index",
                          "preprocess": scorer.preprocess_configuration, "crop_batch_size": scorer.batch_size},
    }
    result["configuration"].update(
        hybrid_weight=.15 if hybrid else 0.,
        hybrid_formula="raw_native_phrase_plus_original_native_absolute_0.15_plus_0.15_contrast_pool_minmax",
        hybrid_minmax_ties="zero", hybrid_failure_policy="explicit_original_selection_fallback",
    )
    keys = ["B", "C", *(["D"] if geometry else [])]
    try:
        scores = scorer.score(image_bgr, boxes, target, references) if len(boxes) else {"records": [], "texts": [],
                     "encoded_crop_count": 0, "image_encode_batch_count": 0}
        records = scores.pop("records")
        result.update(scores)
        for record in records:
            record["geometry"] = _soft_geometry(record["box_xyxy"], image_bgr.shape[:2], plan, anchors) if geometry else 0.
            record["geometry_score"] = (record["contrast_score"] + GEOMETRY_WEIGHT * record["geometry"]
                                        if record["contrast_score"] is not None else None)
        result["crop_scores"] = records
        result["score_status"] = "succeeded" if len(boxes) else "empty_candidates"
        for arm, key in zip(keys, ("target_score", "contrast_score", "geometry_score")):
            chosen = _selected(records, key, baseline_index)
            result["arms"][arm] = {"selected_index": chosen,
                                   "status": "succeeded" if chosen is not None else "no_valid_crop"}
    except CropIdentityError as failure:
        result["score_status"] = failure.code
        for arm in keys:
            result["arms"][arm] = {"selected_index": None, "status": failure.code}
    if hybrid:
        records = result["crop_scores"]
        available = [record for record in records if record["contrast_score"] is not None]
        if baseline_scores is not None and phrase_scores is not None and len(available) == len(boxes) and len(boxes):
            low, high = min(record["contrast_score"] for record in available), max(record["contrast_score"] for record in available)
            span = high - low
            for record in records:
                correction = (record["contrast_score"] - low) / span if span > TIE_EPSILON else 0.
                record["contrast_pool_minmax"] = float(correction)
                record["hybrid_score"] = float(baseline_scores[record["index"]] + .15 * correction)
            result["arms"]["E"] = {"selected_index": _selected(records, "hybrid_score", baseline_index), "status": "succeeded"}
        else:
            reason = ("missing_original_phrase_scores" if phrase_scores is None or baseline_scores is None
                      else "incomplete_valid_crop_pool" if result["score_status"] == "succeeded"
                      else result["score_status"])
            result["arms"]["E"] = {"selected_index": baseline_index, "status": "original_selection_fallback",
                                    "fallback_reason": reason}
    for arm in result["arms"].values():
        index = arm["selected_index"]
        arm["selected_box_xyxy"] = boxes[index].tolist() if index is not None else None
    return result
