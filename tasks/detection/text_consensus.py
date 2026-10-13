"""Conservative role-text agreement on the original detector candidate pool."""
from __future__ import annotations
from typing import Any


def consensus_selection(baseline: int | None, first: int | None, second: int | None,
                        records: list[dict[str, Any]]) -> tuple[int | None, str]:
    """Change identity only if both source-preserving conditions and CLIP agree.

    A new winner must improve both target-only cosine and target-v-reference
    contrast. All boxes still originate from the original complete description.
    No detector confidence, threshold tuning, labels or new candidate is used.
    """
    if (baseline is None or type(first) is not int or type(second) is not int
            or first != second or first == baseline):
        return baseline, "text_no_agreement"
    if not 0 <= baseline < len(records) or not 0 <= first < len(records):
        return baseline, "text_incomplete_identity"
    prior, candidate = records[baseline], records[first]
    for key in ("target_score", "contrast_score"):
        old, new = prior.get(key), candidate.get(key)
        if (not isinstance(old, (float, int)) or not isinstance(new, (float, int))
                or not float("-inf") < old < float("inf") or not float("-inf") < new < float("inf")
                or new <= old + 1e-12):
            return baseline, "text_identity_not_confirmed"
    return first, "text_consensus_selected"
