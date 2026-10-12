"""Offline candidate grounding with semantics scored in one full-image forward.

References decide an optional view before target inference. Actual candidates
from all views are then scored by the full-image target phrase map; confidence
is diagnostic only. No annotations, API calls or model internals enter here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .reference_plan import ReferencePlan
from .reference_grounding import (
    _absolute_region, _box_iou, _expanded_roi, _relation_region, _restore_record, _value,
)


_SOURCE_PRIORITY = {"full_subject": 0, "full_original": 1, "roi": 2}
_GEOMETRY_WEIGHT = .15
_SELF_MATCH_PENALTY = .15
_SELF_MATCH_IOU = .8
_DUPLICATE_IOU = .8
_CROP_FULL_IOU = .5
_CROP_EDGE_TOLERANCE = 2.
_TIE_EPSILON = 1e-12


def _score_array(value: Any, count: int) -> tuple[list[float] | None, str]:
    if value is None:
        return None, "missing"
    try:
        score = np.asarray(value, dtype=np.float64)
        if score.shape != (count,) or not np.isfinite(score).all():
            return None, "invalid"
        return score.tolist(), "valid"
    except (TypeError, ValueError, OverflowError):
        return None, "invalid"


def _capture(detector: Any, image: np.ndarray, description: str,
             extras: list[list[float]] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "description": description, "candidate_boxes": [], "scores": [], "candidate_count": 0,
        "selected_index": None, "selected_box": None, "status": "prediction_failed",
        "phrase_scores": None, "phrase_scores_status": "missing",
        "extra_phrase_scores": None, "extra_phrase_scores_status": "missing",
        "extra_boxes_xyxy": extras,
    }
    kwargs: dict[str, Any] = {"return_phrase_scores": True}
    if extras is not None:
        kwargs["score_boxes_xyxy"] = np.asarray(extras, dtype=np.float64).reshape(-1, 4)
    try:
        prediction = detector.predict(image, description, **kwargs)
    except Exception:
        return record
    try:
        boxes = np.asarray(_value(prediction, "boxes"), dtype=np.float64)
        scores = np.asarray(_value(prediction, "scores"), dtype=np.float64)
        if (boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),)
                or tuple(_value(prediction, "original_size")) != image.shape[:2]
                or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
            record["status"] = "invalid_result"
            return record
        height, width = image.shape[:2]
        boxes = np.clip(boxes, 0, [width, height, width, height])
        record.update(candidate_boxes=boxes.tolist(), scores=scores.tolist(), candidate_count=len(boxes))
        record["phrase_scores"], record["phrase_scores_status"] = _score_array(
            _value(prediction, "phrase_scores"), len(boxes))
        if extras is not None:
            record["extra_phrase_scores"], record["extra_phrase_scores_status"] = _score_array(
                _value(prediction, "extra_phrase_scores"), len(extras))
        if not len(boxes):
            record["status"] = "empty_candidates"
            return record
        index = _value(prediction, "grounding_index")
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not 0 <= int(index) < len(boxes)):
            record["status"] = "invalid_match"
            return record
        record["selected_index"] = int(index)
        selected = boxes[int(index)]
        if not _valid_box(selected):
            record["status"] = "empty_selected_box"
            return record
        record.update(selected_box=selected.tolist(), status="succeeded")
        return record
    except (TypeError, ValueError, OverflowError):
        record["status"] = "invalid_result"
        return record


def _valid_box(box: list[float] | np.ndarray) -> bool:
    return len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def _reliability(reference: dict[str, Any]) -> tuple[float, str]:
    if reference["selected_box"] is None:
        return 0., "reference_unavailable"
    if reference["candidate_count"] == 1:
        return 1., "single_candidate"
    scores = reference["phrase_scores"]
    if scores is None:
        return 0., "phrase_scores_unavailable"
    highest, lowest = max(scores), min(scores)
    span = highest - lowest
    if span <= _TIE_EPSILON:
        return 0., "tied_phrase_scores"
    selected_score = scores[reference["selected_index"]]
    if highest - selected_score > _TIE_EPSILON:
        return 0., "selected_not_raw_best"
    normalized = sorted(((score - lowest) / span for score in scores), reverse=True)
    margin = normalized[0] - normalized[1]
    return float(margin), "unique_phrase_margin" if margin > _TIE_EPSILON else "tied_best_phrase_scores"


def _absolute_score(box: list[float], position: tuple[int | None, int | None] | None,
                    width: int, height: int) -> float:
    if position is None:
        return 0.
    coordinates = ((box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height))
    axes = []
    for coordinate, direction in zip(coordinates, position):
        if direction is None:
            continue
        coordinate = float(np.clip(coordinate, 0, 1))
        axes.append(1 - coordinate if direction == -1 else coordinate if direction == 1
                    else 1 - 2 * abs(coordinate - .5))
    return float(sum(axes) / len(axes)) if axes else 0.


def _relation_score(box: list[float], reference: dict[str, Any], width: int, height: int) -> float:
    anchor = reference["selected_box"]
    if anchor is None or reference["reliability"] <= 0:
        return 0.
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ax, ay = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    relation = reference["relation"]
    if relation == "left_of":
        displacement = (ax - cx) / width
    elif relation == "right_of":
        displacement = (cx - ax) / width
    elif relation == "above":
        displacement = (ay - cy) / height
    elif relation == "below":
        displacement = (cy - ay) / height
    elif relation == "near":
        dx, dy = max(anchor[0] - cx, 0., cx - anchor[2]), max(anchor[1] - cy, 0., cy - anchor[3])
        return 1 - float(np.clip(math.hypot(dx, dy) / (.25 * math.hypot(width, height)), 0, 1))
    else:
        return 0.
    return float(np.clip(.5 + displacement, 0, 1))


def _geometry(box: list[float], references: list[dict[str, Any]], plan: ReferencePlan,
              width: int, height: int) -> tuple[float, list[float], float, bool]:
    relation_scores = [_relation_score(box, reference, width, height) for reference in references]
    absolute_score = _absolute_score(box, plan.absolute_position, width, height)
    denominator = len(plan.references) + (1 if plan.absolute_position is not None else 0)
    weighted = sum(score * reference["reliability"] for score, reference in zip(relation_scores, references))
    geometry = (weighted + absolute_score) / denominator if denominator else 0.
    self_match = any(reference["selected_box"] is not None and reference["reliability"] > 0
                     and _box_iou(box, reference["selected_box"]) >= _SELF_MATCH_IOU
                     for reference in references)
    return float(geometry), relation_scores, absolute_score, self_match


def _touches_internal_crop(box: list[float], roi: list[int], width: int, height: int) -> bool:
    return ((roi[0] > 0 and abs(box[0] - roi[0]) <= _CROP_EDGE_TOLERANCE)
            or (roi[1] > 0 and abs(box[1] - roi[1]) <= _CROP_EDGE_TOLERANCE)
            or (roi[2] < width and abs(box[2] - roi[2]) <= _CROP_EDGE_TOLERANCE)
            or (roi[3] < height and abs(box[3] - roi[3]) <= _CROP_EDGE_TOLERANCE))


def _suppress(pool: list[dict[str, Any]], roi: list[int] | None, width: int, height: int) -> None:
    full = [entry for entry in pool if entry["source"] != "roi" and _valid_box(entry["box"])]
    for entry in pool:
        if not _valid_box(entry["box"]):
            entry.update(suppressed=True, suppression_reason="invalid_area")
        elif entry["semantic_raw"] is None:
            entry.update(suppressed=True, suppression_reason="phrase_score_unavailable")
        elif entry["source"] == "roi" and roi is not None and _touches_internal_crop(entry["box"], roi, width, height):
            overlapping = next((other for other in full if other["semantic_raw"] is not None
                                and _box_iou(entry["box"], other["box"]) >= _CROP_FULL_IOU), None)
            if overlapping is not None:
                entry.update(suppressed=True, suppression_reason="roi_internal_boundary_full_overlap",
                             suppression_partner=overlapping["pool_index"])
    ordered = sorted(pool, key=lambda entry: (_SOURCE_PRIORITY[entry["source"]],
                     -(entry["semantic_raw"] if entry["semantic_raw"] is not None else -math.inf), entry["source_index"]))
    retained = []
    for entry in ordered:
        if entry["suppressed"]:
            continue
        duplicate = next((other for other in retained if _box_iou(entry["box"], other["box"]) >= _DUPLICATE_IOU), None)
        if duplicate is None:
            retained.append(entry)
        else:
            entry.update(suppressed=True, suppression_reason="duplicate_iou",
                         suppression_partner=duplicate["pool_index"])


def _choose(entries: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(entries, key=lambda entry: (entry[field], -_SOURCE_PRIORITY[entry["source"]], -entry["source_index"]))


def predict_candidate_grounding(detector: Any, image_bgr: np.ndarray,
                                plan: ReferencePlan) -> dict[str, Any]:
    """Return all candidates and bounded semantic/geometry selection diagnostics.

    The detector must support ``return_phrase_scores`` and ``score_boxes_xyxy``.
    Only the last full-subject forward's raw phrase scores enter target ranking.
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3
            or min(image_bgr.shape[:2]) <= 0):
        raise ValueError("image_bgr must be a nonempty HxWx3 array")
    if any(not isinstance(text, str) or not text.strip()
           for text in (plan.original_description, plan.target_description)):
        raise ValueError("Original and target descriptions must be nonempty")
    height, width = image_bgr.shape[:2]
    absolute = _absolute_region(plan.absolute_position, width, height)
    region = list(absolute)
    result: dict[str, Any] = {
        "strategy_version": 2, "input_hw": [height, width], "references": [],
        "roi_record": None, "full_original": None, "full_subject": None, "call_order": [],
        "search_region_xyxy": None, "pool": [], "pool_candidate_count": 0, "active_candidate_count": 0,
        "suppressed_candidate_count": 0, "selected_box": None, "selected_source": "none",
        "selected_pool_index": None, "unconstrained_selected_box": None,
        "unconstrained_selected_pool_index": None, "selection_reason": "no_valid_target",
        "fallback_reason": None, "roi_fallback_reason": None,
        "configuration": {"geometry_weight": _GEOMETRY_WEIGHT, "self_match_penalty": _SELF_MATCH_PENALTY,
                          "self_match_iou": _SELF_MATCH_IOU, "duplicate_iou": _DUPLICATE_IOU,
                          "crop_full_overlap_iou": _CROP_FULL_IOU, "crop_edge_tolerance_pixels": _CROP_EDGE_TOLERANCE,
                          "near_distance_fraction_diagonal": .25, "roi_padding_fraction": .10,
                          "minimum_roi_fraction": .25, "score_tie_epsilon": _TIE_EPSILON,
                          "semantic_normalization": "active_pool_minmax_ties_zero",
                          "geometry_denominator": "parsed_reference_count_plus_absolute_constraint",
                          "source_priority": list(_SOURCE_PRIORITY)},
    }

    def call(kind: str, image: np.ndarray, description: str, roi: list[int] | None = None,
             extras: list[list[float]] | None = None) -> dict[str, Any]:
        result["call_order"].append({"kind": kind, "description": description,
                                     "input_hw": list(image.shape[:2]), "roi_xyxy": roi,
                                     "extra_box_count": len(extras) if extras is not None else 0})
        return _capture(detector, image, description, extras)

    roi_reason = None if plan.references else "no_references"
    for spec in plan.references:
        reference = call("reference", image_bgr, spec.description)
        reference.update(relation=spec.relation)
        reliability, reliability_reason = _reliability(reference)
        reference.update(reliability=reliability, reliability_reason=reliability_reason)
        result["references"].append(reference)
        if reliability <= _TIE_EPSILON:
            roi_reason = "reference_unreliable"
            continue
        try:
            constraint = _relation_region(reference["selected_box"], spec.relation, width, height)
        except ValueError:
            reference.update(reliability=0., reliability_reason="unsupported_relation")
            roi_reason = "unsupported_relation"
            continue
        region = [max(region[0], constraint[0]), max(region[1], constraint[1]),
                  min(region[2], constraint[2]), min(region[3], constraint[3])]
    roi = None
    if roi_reason is None:
        result["search_region_xyxy"] = region
        if region[2] <= region[0] or region[3] <= region[1]:
            roi_reason = "conflicting_search_region"
        else:
            roi = _expanded_roi(region, width, height)
            if roi == [0, 0, width, height]:
                roi, roi_reason = None, "whole_frame_roi"
            elif roi[2] <= roi[0] or roi[3] <= roi[1]:
                roi, roi_reason = None, "empty_roi"
            else:
                crop = np.ascontiguousarray(image_bgr[roi[1]:roi[3], roi[0]:roi[2]])
                roi_record = call("target_roi", crop, plan.target_description, roi)
                _restore_record(roi_record, roi, width, height)
                roi_record["roi_xyxy"] = roi
                result["roi_record"] = roi_record
                if not roi_record["candidate_count"]:
                    roi_reason = "roi_prediction_unavailable"
    result["roi_fallback_reason"] = roi_reason
    full_original = call("target_full_original", image_bgr, plan.original_description)
    result["full_original"] = full_original
    extras: list[list[float]] = []
    extra_sources: list[tuple[str, int]] = []
    for source, record in (("roi", result["roi_record"]), ("full_original", full_original)):
        if record is None:
            continue
        for index, box in enumerate(record["candidate_boxes"]):
            if _valid_box(box):
                extras.append(list(box))
                extra_sources.append((source, index))
    full_subject = call("target_full_subject", image_bgr, plan.target_description, extras=extras)
    full_subject["extra_candidate_sources"] = [list(source) for source in extra_sources]
    result["full_subject"] = full_subject
    extra_scores = full_subject["extra_phrase_scores"]
    score_lookup = {(source, index): (extra_scores[extra_index] if extra_scores is not None else None, extra_index)
                    for extra_index, (source, index) in enumerate(extra_sources)}
    for source, record in (("full_subject", full_subject), ("full_original", full_original), ("roi", result["roi_record"])):
        if record is None:
            continue
        for index, box in enumerate(record["candidate_boxes"]):
            raw, extra_index = score_lookup.get((source, index), (None, None))
            if source == "full_subject":
                raw = record["phrase_scores"][index] if record["phrase_scores"] is not None else None
            geometry, relation_scores, absolute_score, self_match = _geometry(
                box, result["references"], plan, width, height)
            result["pool"].append({"pool_index": len(result["pool"]), "source": source, "source_index": index,
                "box": list(box), "semantic_raw": raw, "semantic_norm": None, "geometry": geometry,
                "relation_scores": relation_scores, "absolute_score": absolute_score, "self_match": self_match,
                "self_match_penalty": _SELF_MATCH_PENALTY if self_match else 0., "total_score": None,
                "scoring_origin": "full_subject_same_forward", "extra_score_index": extra_index,
                "suppressed": False, "suppression_reason": None, "suppression_partner": None})
    pool = result["pool"]
    _suppress(pool, roi, width, height)
    active = [entry for entry in pool if not entry["suppressed"]]
    result.update(pool_candidate_count=len(pool), active_candidate_count=len(active),
                  suppressed_candidate_count=len(pool) - len(active))
    if active:
        minimum, maximum = min(entry["semantic_raw"] for entry in active), max(entry["semantic_raw"] for entry in active)
        span = maximum - minimum
        for entry in active:
            entry["semantic_norm"] = ((entry["semantic_raw"] - minimum) / span if span > _TIE_EPSILON else 0.)
            entry["total_score"] = (entry["semantic_norm"] + _GEOMETRY_WEIGHT * entry["geometry"]
                                    - entry["self_match_penalty"])
        unconstrained, selected = _choose(active, "semantic_norm"), _choose(active, "total_score")
        result.update(unconstrained_selected_box=unconstrained["box"],
                      unconstrained_selected_pool_index=unconstrained["pool_index"],
                      selected_box=selected["box"], selected_source=selected["source"],
                      selected_pool_index=selected["pool_index"], selection_reason="candidate_score")
    else:
        result["fallback_reason"] = "phrase_scores_unavailable"
        for source, record in (("full_subject", full_subject), ("full_original", full_original)):
            if record["selected_box"] is not None:
                result.update(selected_box=record["selected_box"], selected_source=source,
                              selection_reason=source + "_match_fallback")
                break
    return result
