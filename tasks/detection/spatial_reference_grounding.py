"""Locate positioned references, derive a region, then match the target there.

Uses the submitted detector and its phrase-map assignment only. Reference boxes
constrain the search; they never become target boxes. No crop-image CLIP scores.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
import time
from typing import Any, Callable

import numpy as np
import torch

from text_encoder.matching import assign_by_score_matrix
from .reference_plan import (ReferenceSpec, parse_reference_plan, _absolute_position,
                             _EN_ABS_PREFIX, _EN_ABS_SUFFIX, _ZH_ABS_PREFIX, _ZH_ABS_SUFFIX)
from .reference_grounding import (_absolute_region, _relation_region, _expanded_roi,
                                 _box_iou)
# Reuse only source-bound grammar, never the experimental mixed encoder.
from .text_conditioning import parse_grounding_text


@dataclass(frozen=True)
class SpatialReferencePlan:
    target_query: str
    references: tuple[ReferenceSpec, ...]
    absolute_position: tuple[int | None, int | None] | None


_ZH_ATTRIBUTE = re.compile(
    r"^(?:它|该目标|这个目标|该物体)(?:是|呈|呈现|为)(?:一个|一片|一块)?"
    r"((?:梯形|矩形|长方形|正方形|圆形|三角形|椭圆形|绿色|灰色|白色|红色|蓝色|黑色)(?:的)?)$"
)
_EN_LOCATION = re.compile(
    r"^(?:it|the target|this target|the object)\s+is\s+"
    r"(?:(?:located|situated)\s+)?((?:in|at|on)\s+.+)$", re.I,
)


def _identity(text: str) -> str:
    """Remove explicit global position only, retaining source object attributes."""
    parts = text.split(",")
    first = parts[0].strip()
    for pattern in (_EN_ABS_SUFFIX, _ZH_ABS_SUFFIX):
        match = pattern.search(first)
        if match:
            first = first[:match.start()].strip()
    match = _ZH_ABS_PREFIX.match(first)
    if match:
        first = first[match.end():].strip()
    else:
        match = _EN_ABS_PREFIX.match(first)
        if match and re.search(r"[\s-]", match.group("position")):
            start, end = match.span("position")
            tail = re.sub(r"^\s+(?:corner|region|area|side)\b", "", first[end:], flags=re.I)
            first = first[:start] + tail
    return ", ".join([" ".join(first.split()), *[part.strip() for part in parts[1:]]]).strip()


def parse_spatial_reference_plan(description: str) -> SpatialReferencePlan | None:
    """Account for explicit attributes/location sentences before relation parsing."""
    plan = parse_grounding_text(description)
    if plan is None:
        clauses = [p.strip() for p in re.split(r"[。.;；\n]+", description) if p.strip()]
        if not clauses:
            return None
        first, rest, attributes = clauses[0], [], []
        for clause in clauses[1:]:
            attribute = _ZH_ATTRIBUTE.fullmatch(clause)
            location = _EN_LOCATION.fullmatch(clause)
            if attribute:
                attributes.append(attribute.group(1))
            elif location and _absolute_position("object " + location.group(1))[1] is not None:
                first += " " + location.group(1)
            else:
                # Explicit target pronouns do not change relation attachment.
                clause = re.sub(r"^(?:它|该目标|这个目标)(?=位于|处于|在|靠近|紧邻)", "", clause)
                clause = re.sub(r"^(it\s+is)\s+(?:located|situated)\s+", r"\1 ", clause, flags=re.I)
                rest.append(clause)
        relational = parse_reference_plan(", ".join([first, *rest]))
        if relational is None:
            return None
        target = relational.target_description
        if attributes:
            target += ", " + ", ".join(attributes)
        references = relational.references
    else:
        target, references = plan.target_description, plan.references
    if not references:
        return None
    valid, position = _absolute_position(target.split(",")[0])
    if not valid:
        return None
    query = _identity(target) if position is not None else target
    if not query or query.casefold() in {"a", "an", "the"}:
        return None
    return SpatialReferencePlan(query, references, position)


def positioned_reference(spec: ReferenceSpec, position: tuple | None) -> tuple[str, tuple | None]:
    """Carry target coarse location; preserve an explicit reference's own location."""
    valid, explicit = _absolute_position(spec.description)
    if not valid:
        raise ValueError("Conflicting reference location")
    if explicit is not None or position is None:
        return spec.description, explicit
    labels = {
        (-1, -1): ("top-left corner", "左上角"), (1, -1): ("top-right corner", "右上角"),
        (-1, 1): ("bottom-left corner", "左下角"), (1, 1): ("bottom-right corner", "右下角"),
        (-1, 0): ("middle-left region", "左侧中央"), (1, 0): ("middle-right region", "右侧中央"),
        (0, -1): ("top center", "上方中央"), (0, 1): ("bottom center", "下方中央"),
        (0, 0): ("center", "中央"), (-1, None): ("left side", "左侧"),
        (1, None): ("right side", "右侧"), (None, -1): ("top", "上方"),
        (None, 1): ("bottom", "下方"),
    }
    english, chinese = labels[position]
    query = (chinese + "的" + spec.description if re.search(r"[\u3400-\u9fff]", spec.description)
             else spec.description + " in the " + english + " of the image")
    return query, position


def _candidates(result: Any, hw: tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(result.boxes, dtype=np.float64)
    scores = np.asarray(result.scores)
    height, width = hw
    if (tuple(result.original_size) != hw or boxes.ndim != 2 or boxes.shape[1] != 4
            or scores.shape != (len(boxes),) or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all() or (boxes < 0).any()
            or (boxes > [width, height, width, height]).any()):
        raise ValueError("Invalid native candidates")
    return boxes


def _match(result: Any, boxes: np.ndarray, region: list[float], *,
           adjusted: bool, anchors: list[list[float]] = ()) -> int | None:
    """Same one-row phrase-map Hungarian match, restricted by global geometry."""
    row = result.baseline_adjusted_scores if adjusted else result.phrase_scores
    if row is None:
        raise ValueError("Missing native phrase scores")
    row = np.asarray(row, dtype=np.float32)
    if row.shape != (len(boxes),) or not np.isfinite(row).all():
        raise ValueError("Invalid native phrase scores")
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    valid = ((centers >= region[:2]).all(axis=1) & (centers <= region[2:]).all(axis=1)
             & (boxes[:, 2:] > boxes[:, :2]).all(axis=1))
    for anchor in anchors:
        valid &= np.asarray([_box_iou(box.tolist(), anchor) < .8 for box in boxes], dtype=bool)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    rows, columns = assign_by_score_matrix(torch.from_numpy(row[indices].copy()).unsqueeze(0), temperature=.7)
    if rows.numel() != 1 or columns.numel() != 1 or int(rows[0]) != 0:
        raise ValueError("Invalid native assignment")
    return int(indices[int(columns[0])])


class SpatialReferenceGroundingExecutor:
    def __init__(self, native: Any, *, description_extractor: Callable,
                 image_converter: Callable) -> None:
        if (not callable(getattr(native, "predict", None)) or not callable(description_extractor)
                or not callable(image_converter)):
            raise ValueError("spatial_reference requires native detection and input preparation")
        self.native, self.description_extractor, self.image_converter = native, description_extractor, image_converter

    def predict(self, image: Any, original_question: str) -> Any:
        started = time.monotonic()
        description = self.description_extractor(original_question)
        plan = parse_spatial_reference_plan(description)
        reference_count, located, roi_count, calls = len(plan.references) if plan else 0, 0, 0, 0

        def finished(result: Any, reason: str, selected: bool = False) -> Any:
            diagnostics = dict(result.diagnostics or {})
            diagnostics.update(
                grounding_policy="spatial_reference", grounding_parse_status="parsed" if plan else "unsupported",
                grounding_outcome="region_selected" if selected else "native_fallback", grounding_reason=reason,
                grounding_reference_count=reference_count, grounding_located_reference_count=located,
                grounding_candidate_count=len(result.boxes), grounding_roi_count=roi_count,
                grounding_detector_call_count=calls,
                grounding_elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return replace(result, diagnostics=diagnostics)

        reason = "grounding_unsupported_plan"
        if plan is not None:
            original = self.image_converter(image)
            height, width = original.shape[:2]
            region = _absolute_region(plan.absolute_position, width, height)
            anchors = []
            reason = "grounding_reference_unavailable"
            try:
                for spec in plan.references:
                    query, position = positioned_reference(spec, plan.absolute_position)
                    calls += 1
                    reference = self.native.predict(original, query, return_phrase_scores=True)
                    boxes = _candidates(reference, (height, width))
                    index = _match(reference, boxes, _absolute_region(position, width, height), adjusted=True)
                    if index is None:
                        break
                    anchor = boxes[index].tolist()
                    anchors.append(anchor)
                    located += 1
                    constraint = _relation_region(anchor, spec.relation, width, height)
                    region = [max(region[0], constraint[0]), max(region[1], constraint[1]),
                              min(region[2], constraint[2]), min(region[3], constraint[3])]
                if located == reference_count:
                    reason = "grounding_conflicting_region"
                    if region[2] > region[0] and region[3] > region[1]:
                        roi = _expanded_roi(region, width, height)
                        crop = np.ascontiguousarray(original[roi[1]:roi[3], roi[0]:roi[2]])
                        roi_count, calls = 1, calls + 1
                        reason = "grounding_roi_unavailable"
                        # Global position is already enforced in original coordinates.
                        # Native raw phrase scores avoid a crop-frame spatial bonus.
                        target = self.native.predict(crop, plan.target_query, require_grounding=False,
                                                     return_phrase_scores=True)
                        boxes = _candidates(target, tuple(crop.shape[:2]))
                        boxes = boxes + [roi[0], roi[1], roi[0], roi[1]]
                        index = _match(target, boxes, region, adjusted=False, anchors=anchors)
                        if index is not None:
                            target = replace(target, boxes=boxes.astype(np.float32), original_size=(height, width),
                                             grounding_index=index)
                            return finished(target, "grounding_region_selected", True)
            except Exception:
                # Do not log input text, coordinates, paths or exception strings.
                # The mandatory original native fallback still propagates failure.
                pass
        calls += 1
        return finished(self.native.predict(image, original_question), reason)
