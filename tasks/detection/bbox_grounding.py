"""Optional bbox identity selection over the unchanged native candidate pool.

The API supplies input preparation and the existing native runtime. This task
module does not import API code or construct a detector or CLIP model.
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Callable

from .crop_identity import rank_full_candidates
from .reference_plan import parse_reference_plan


class BBoxGroundingExecutor:
    """Parse raw descriptions first, then apply the frozen guarded E policy."""

    def __init__(self, native: Any, scorer: Any, *,
                 description_extractor: Callable[[str], str],
                 image_converter: Callable[[Any], Any]) -> None:
        self._validate_native(native)
        if (not callable(getattr(native, "predict", None)) or not callable(getattr(scorer, "score", None))
                or not callable(description_extractor) or not callable(image_converter)):
            raise ValueError("clip_guarded requires initialized native, scorer and input preparation")
        self.native = native
        self.scorer = scorer
        self.description_extractor = description_extractor
        self.image_converter = image_converter

    @staticmethod
    def _validate_native(native: Any) -> None:
        if native.candidate_confidence != .25 or native.spatial_weight != .15:
            raise ValueError("clip_guarded requires native confidence .25 and spatial weight .15")

    @staticmethod
    def _result(native_result: Any, selected: int | None, *, started: float,
                parse_status: str, reference_count: int, reason: str,
                crop_count: int = 0) -> Any:
        diagnostics = dict(native_result.diagnostics or {})
        diagnostics.update(
            grounding_policy="clip_guarded", grounding_parse_status=parse_status,
            grounding_outcome="hybrid_selected" if reason == "grounding_hybrid_selected" else "native_fallback",
            grounding_reason=reason, grounding_reference_count=reference_count,
            grounding_candidate_count=len(native_result.boxes), grounding_crop_count=crop_count,
            grounding_elapsed_ms=round((time.monotonic() - started) * 1000),
        )
        return replace(native_result, grounding_index=selected, diagnostics=diagnostics)

    def predict(self, image: Any, original_question: str) -> Any:
        started = time.monotonic()
        self._validate_native(self.native)
        # Newlines delimit reference clauses. Only the native encoder sanitizes
        # its own input; parsing must see the raw extracted description first.
        description = self.description_extractor(original_question)
        plan = parse_reference_plan(description)
        if plan is None:
            result = self.native.predict(image, original_question)
            return self._result(result, result.grounding_index, started=started,
                                parse_status="unsupported", reference_count=0,
                                reason="grounding_unsupported_plan")
        # Native inference failures propagate. Only optional identity ranking
        # may fall back, and always to this same forward's original selection.
        result = self.native.predict(image, original_question, return_phrase_scores=True)
        if not len(result.boxes):
            return self._result(result, result.grounding_index, started=started,
                                parse_status="parsed", reference_count=len(plan.references),
                                reason="grounding_empty_candidates")
        if result.phrase_scores is None or result.baseline_adjusted_scores is None:
            return self._result(result, result.grounding_index, started=started,
                                parse_status="parsed", reference_count=len(plan.references),
                                reason="grounding_missing_scores")
        try:
            ranking = rank_full_candidates(
                self.scorer, self.image_converter(image), description,
                result.boxes, result.grounding_index, phrase_scores=result.phrase_scores,
                hybrid=True, baseline_scores=result.baseline_adjusted_scores,
            )
            arm = ranking["arms"]["E"]
            selected = arm["selected_index"]
            crop_count = ranking.get("encoded_crop_count", 0)
            crop_count = crop_count if type(crop_count) is int and crop_count >= 0 else 0
            if arm["status"] != "succeeded":
                reason = "grounding_crop_failed"
                selected = result.grounding_index
            elif type(selected) is not int or not 0 <= selected < len(result.boxes):
                reason = "grounding_invalid_selection"
                selected = result.grounding_index
            else:
                reason = "grounding_hybrid_selected"
        except Exception:
            selected, crop_count, reason = result.grounding_index, 0, "grounding_rank_failed"
        return self._result(result, selected, started=started, parse_status="parsed",
                            reference_count=len(plan.references), reason=reason, crop_count=crop_count)
