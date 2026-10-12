"""Reference-first public grounding experiment with unchanged model boundaries.

An injected detector supplies its own description-matched ``grounding_index``.
This module chooses a search view from references before any target inference;
it never uses annotations, detector confidence, or fabricated target boxes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .reference_plan import ReferencePlan


_RELATIONS = frozenset({"left_of", "right_of", "above", "below", "near"})
_PADDING = .10
_MINIMUM_SIZE = .25
_NEAR_EXPANSION = .25


def _value(result: Any, name: str) -> Any:
    return result.get(name) if isinstance(result, dict) else getattr(result, name, None)


def _detector_record(detector: Any, image: np.ndarray, description: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "candidate_boxes": [], "scores": [], "candidate_count": 0,
        "selected_index": None, "selected_box": None, "status": "prediction_failed",
    }
    try:
        result = detector.predict(image, description)
    except Exception:
        # Error messages can include paths or input text; only keep a fixed status.
        return record
    try:
        boxes = np.asarray(_value(result, "boxes"), dtype=np.float64)
        scores = np.asarray(_value(result, "scores"), dtype=np.float64)
        original_size = tuple(_value(result, "original_size"))
        if (boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),)
                or original_size != image.shape[:2]
                or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
            record["status"] = "invalid_result"
            return record
        height, width = image.shape[:2]
        boxes = np.clip(boxes, 0, [width, height, width, height])
        record.update(candidate_boxes=boxes.tolist(), scores=scores.tolist(), candidate_count=len(boxes))
        if not len(boxes):
            record["status"] = "empty_candidates"
            return record
        index = _value(result, "grounding_index")
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not 0 <= int(index) < len(boxes)):
            record["status"] = "invalid_match"
            return record
        record["selected_index"] = int(index)
        selected = boxes[int(index)]
        if selected[2] <= selected[0] or selected[3] <= selected[1]:
            record["status"] = "empty_selected_box"
            return record
        record.update(selected_box=selected.tolist(), status="succeeded")
        return record
    except (TypeError, ValueError, OverflowError):
        record["status"] = "invalid_result"
        return record


def _absolute_region(position: tuple[int | None, int | None] | None,
                     width: int, height: int) -> list[float]:
    region = [0., 0., float(width), float(height)]
    if position is None:
        return region
    if len(position) != 2:
        raise ValueError("absolute_position must have two axes")
    for axis, direction in enumerate(position):
        extent = float(width if axis == 0 else height)
        if direction is None:
            continue
        if type(direction) is not int or direction not in (-1, 0, 1):
            raise ValueError("absolute_position axes must be -1, 0, 1 or None")
        if direction == -1:
            region[axis + 2] = .6 * extent
        elif direction == 1:
            region[axis] = .4 * extent
        else:
            region[axis], region[axis + 2] = .2 * extent, .8 * extent
    return region


def _relation_region(box: list[float], relation: str, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if relation == "left_of":
        return [0., 0., cx, float(height)]
    if relation == "right_of":
        return [cx, 0., float(width), float(height)]
    if relation == "above":
        return [0., 0., float(width), cy]
    if relation == "below":
        return [0., cy, float(width), float(height)]
    if relation == "near":
        return [max(0., x1 - _NEAR_EXPANSION * width), max(0., y1 - _NEAR_EXPANSION * height),
                min(float(width), x2 + _NEAR_EXPANSION * width),
                min(float(height), y2 + _NEAR_EXPANSION * height)]
    raise ValueError("Unsupported target-to-reference relation")


def _expanded_roi(region: list[float], width: int, height: int) -> list[int]:
    expanded = [max(0., region[0] - _PADDING * width),
                max(0., region[1] - _PADDING * height),
                min(float(width), region[2] + _PADDING * width),
                min(float(height), region[3] + _PADDING * height)]
    for axis, extent in enumerate((width, height)):
        minimum = _MINIMUM_SIZE * extent
        if expanded[axis + 2] - expanded[axis] < minimum:
            center = (expanded[axis] + expanded[axis + 2]) / 2
            start = min(max(0., center - minimum / 2), extent - minimum)
            expanded[axis], expanded[axis + 2] = start, start + minimum
    return [math.floor(expanded[0]), math.floor(expanded[1]),
            min(width, math.ceil(expanded[2])), min(height, math.ceil(expanded[3]))]


def _restore_record(record: dict[str, Any], roi: list[int], width: int, height: int) -> None:
    offset = np.array([roi[0], roi[1], roi[0], roi[1]], dtype=np.float64)
    if record["candidate_boxes"]:
        boxes = np.clip(np.asarray(record["candidate_boxes"]) + offset, 0, [width, height, width, height])
        record["candidate_boxes"] = boxes.tolist()
    if record["selected_box"] is not None:
        box = np.clip(np.asarray(record["selected_box"]) + offset, 0, [width, height, width, height])
        if box[2] <= box[0] or box[3] <= box[1]:
            record.update(selected_box=None, status="empty_selected_box")
        else:
            record["selected_box"] = box.tolist()


def _box_iou(first: list[float], second: list[float]) -> float:
    intersection = max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(
        0., min(first[3], second[3]) - max(first[1], second[1]))
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.


def _proposal_checks(box: list[float], references: list[dict[str, Any]],
                     absolute_region: list[float], width: int, height: int) -> tuple[bool, bool]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    valid = (absolute_region[0] <= cx <= absolute_region[2]
             and absolute_region[1] <= cy <= absolute_region[3])
    self_match = False
    for reference in references:
        anchor = reference["selected_box"]
        ax, ay = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
        relation = reference["relation"]
        if relation == "left_of":
            consistent = cx < ax
        elif relation == "right_of":
            consistent = cx > ax
        elif relation == "above":
            consistent = cy < ay
        elif relation == "below":
            consistent = cy > ay
        else:
            near = _relation_region(anchor, relation, width, height)
            consistent = near[0] <= cx <= near[2] and near[1] <= cy <= near[3]
        reference["target_relation_satisfied"] = bool(consistent)
        valid = valid and consistent
        self_match = self_match or _box_iou(box, anchor) >= .8
    return bool(valid), bool(self_match)


def predict_reference_grounding(detector: Any, image_bgr: np.ndarray,
                                plan: ReferencePlan) -> dict[str, Any]:
    """Run references, one expanded target view, then the full target baseline.

    All target calls use ``plan.original_description``. Results are plain JSON-
    compatible records with continuous original-image xyxy coordinates. A full
    target prediction is always collected last for fallback and paired scoring.
    Success of the ROI geometry checks does not prove reference identity or
    preserve the baseline's accuracy; public experiments must report regressions.
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) <= 0):
        raise ValueError("image_bgr must be a nonempty HxWx3 array")
    if not isinstance(plan.original_description, str) or not plan.original_description.strip():
        raise ValueError("A nonempty original description is required")
    height, width = image_bgr.shape[:2]
    absolute = _absolute_region(plan.absolute_position, width, height)
    region = list(absolute)
    result: dict[str, Any] = {
        "strategy_version": 1, "input_hw": [height, width], "references": [], "rois": [],
        "full_target": None, "call_order": [], "search_region_xyxy": None,
        "selected_box": None, "selected_source": "none", "selection_reason": "no_valid_target",
        "fallback_reason": None,
        "configuration": {"padding_fraction": _PADDING, "minimum_roi_fraction": _MINIMUM_SIZE,
                          "near_expansion_fraction": _NEAR_EXPANSION, "self_match_iou": .8},
    }
    fallback_reason = "no_references" if not plan.references else None
    for spec in plan.references:
        result["call_order"].append({"kind": "reference", "description": spec.description,
                                     "input_hw": [height, width], "roi_xyxy": None})
        reference = _detector_record(detector, image_bgr, spec.description)
        reference.update(description=spec.description, relation=spec.relation,
                         target_relation_satisfied=None)
        result["references"].append(reference)
        if reference["selected_box"] is None:
            fallback_reason = "reference_unavailable"
        elif spec.relation not in _RELATIONS:
            fallback_reason = "unsupported_relation"
        else:
            constraint = _relation_region(reference["selected_box"], spec.relation, width, height)
            region = [max(region[0], constraint[0]), max(region[1], constraint[1]),
                      min(region[2], constraint[2]), min(region[3], constraint[3])]
    proposal = None
    if fallback_reason is None:
        result["search_region_xyxy"] = region
        if region[2] <= region[0] or region[3] <= region[1]:
            fallback_reason = "conflicting_search_region"
        else:
            roi = _expanded_roi(region, width, height)
            if roi == [0, 0, width, height]:
                fallback_reason = "whole_frame_roi"
            elif roi[2] <= roi[0] or roi[3] <= roi[1]:
                fallback_reason = "empty_roi"
            else:
                crop = np.ascontiguousarray(image_bgr[roi[1]:roi[3], roi[0]:roi[2]])
                result["call_order"].append({"kind": "target_roi", "description": plan.original_description,
                                             "input_hw": list(crop.shape[:2]), "roi_xyxy": roi})
                roi_record = _detector_record(detector, crop, plan.original_description)
                _restore_record(roi_record, roi, width, height)
                roi_record.update(roi_xyxy=roi, valid_relations=False, self_match=False)
                result["rois"].append(roi_record)
                if roi_record["selected_box"] is None:
                    fallback_reason = "roi_prediction_unavailable"
                else:
                    valid, self_match = _proposal_checks(roi_record["selected_box"], result["references"],
                                                         absolute, width, height)
                    roi_record.update(valid_relations=valid, self_match=self_match)
                    if self_match:
                        fallback_reason = "reference_self_match"
                    elif not valid:
                        fallback_reason = "relation_mismatch"
                    else:
                        proposal = roi_record["selected_box"]
    result["call_order"].append({"kind": "target_full", "description": plan.original_description,
                                 "input_hw": [height, width], "roi_xyxy": None})
    full_target = _detector_record(detector, image_bgr, plan.original_description)
    result["full_target"] = full_target
    result["fallback_reason"] = fallback_reason
    if proposal is not None:
        result.update(selected_box=proposal, selected_source="roi", selection_reason="roi_selected")
    elif full_target["selected_box"] is not None:
        result.update(selected_box=full_target["selected_box"], selected_source="full",
                      selection_reason="full_target_fallback")
    return result
