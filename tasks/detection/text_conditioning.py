"""Source-bound target cues alongside the complete grounding description.

The complete description remains in the detector condition. This adapter only
uses explicit target/reference grammar; uncertain attachment is left unchanged.
It reuses the native encoder and never translates, invents objects or reads gold.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

import torch

from .reference_plan import (ReferenceSpec, parse_reference_plan, _atomic_phrase,
                             _NEGATIVE, _UNSUPPORTED, _absolute_position,
                             _EN_ABS_PREFIX, _EN_ABS_SUFFIX, _ABS_POSITIONS)


@dataclass(frozen=True)
class GroundingTextPlan:
    original_description: str
    target_description: str
    references: tuple[ReferenceSpec, ...]
    attributes: tuple[str, ...] = ()


_ATTRIBUTE_WORDS = frozenset((
    "red orange yellow green blue cyan purple pink white gray grey black brown "
    "beige tan silver golden light dark pale bright small large big tiny long "
    "short wide narrow round square rectangular circular triangular trapezoidal "
    "oval curved straight flat pitched concrete asphalt metallic metal wooden "
    "roof roofs rooftop shaped colored coloured and a an the very relatively"
).split())
_ATTRIBUTE_CLAUSE = re.compile(
    r"^(?:it|the target|this target|that target|the object)\s+"
    r"(?:is|has|has a|has an)\s+(.+)$", re.I,
)
_TARGET_COPULA = re.compile(r"^(.+?)\s+is\s+(.+)$", re.I)
_SENTENCES = re.compile(r"[.;\n]+\s*")
_REQUEST = re.compile(r"^(?:please\s+)?(?:locate|find|identify|select|mark|box)\s+", re.I)


def _attribute(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    return bool(words) and all(word in _ATTRIBUTE_WORDS for word in words)


def parse_grounding_text(description: str) -> GroundingTextPlan | None:
    """Keep target attributes and reject unaccounted sentences or attachments."""
    if not isinstance(description, str) or not description.strip():
        return None
    text = _REQUEST.sub("", description.strip()).rstrip(".")
    if _NEGATIVE.search(text) or _UNSUPPORTED.search(text):
        return None
    original = parse_reference_plan(text)
    if original is not None:
        return GroundingTextPlan(description, original.target_description, original.references)
    clauses = [clause.strip() for clause in _SENTENCES.split(text) if clause.strip()]
    if not clauses:
        return None
    attributes: list[str] = []
    first = clauses[0]
    copula = _TARGET_COPULA.fullmatch(first)
    if copula is not None and _attribute(copula.group(2)):
        first = copula.group(1).strip()
        attributes.append(copula.group(2).strip())
    remaining = []
    for clause in clauses[1:]:
        match = _ATTRIBUTE_CLAUSE.fullmatch(clause)
        if match is not None and _attribute(match.group(1)):
            attributes.append(match.group(1).strip())
        else:
            remaining.append(clause)
    if not attributes and len(clauses) == 1 and _atomic_phrase(first):
        return GroundingTextPlan(description, first, ())
    if not attributes:
        return None
    reduced = ", ".join([first, *remaining])
    relational = parse_reference_plan(reduced)
    # Without relations, accept only a single explicit noun phrase and known
    # attributes. An unrelated second sentence is never silently removed.
    if relational is None:
        if (remaining or first.casefold() in {"it", "they", "the target", "the object"}
                or re.search(r"\b(?:is|are|and|or|not|no|without|which|that|near|above|below|left|right)\b", first, re.I)):
            return None
        target, references = first, ()
    else:
        target, references = relational.target_description, relational.references
    return GroundingTextPlan(description, target + ", " + ", ".join(attributes), references, tuple(attributes))


def visual_description(description: str) -> str:
    """Remove explicit image-frame coordinates only from a crop identity cue.

    A crop has its own coordinate frame. The complete source remains the native
    detector prompt and supplies the original-image position prior separately.
    Relative/nested positions and ambiguous attachment are retained verbatim.
    """
    valid, position = _absolute_position(description)
    if not valid or position is None:
        return description
    # The old helper checks the prefix and final suffix. Also inspect earlier
    # frame suffixes so contradictory internal positions cannot be half-erased.
    mentions = []
    for match in re.finditer(_EN_ABS_SUFFIX.pattern[:-1], description, re.I):
        key = " ".join(match.group("position").casefold().replace("-", " ").split())
        mentions.append(_ABS_POSITIONS[key])
    if any(mention != position for mention in mentions):
        return description
    result = description
    suffix = _EN_ABS_SUFFIX.search(result)
    if suffix is not None:
        result = result[:suffix.start()].strip()
    prefix = _EN_ABS_PREFIX.search(result)
    if prefix is not None and re.search(r"[\s-]", prefix.group("position")):
        start, end = prefix.span("position")
        result = result[:start] + result[end:]
    result = " ".join(result.split())
    return result if result and _atomic_phrase(result) else description


class RoleConditionedEncoder:
    """One normalized detector row: 75% complete source + 25% target cue.

    The native encoder checks every full string against its real 75-content-token
    bound. No chunking, truncation or modified CLIP inference is hidden here.
    No global encoder state is changed: counting/presence use their own base.
    """
    def __init__(self, base: Any, *, target_weight: float = .25) -> None:
        if target_weight not in (.25, .5) or isinstance(target_weight, bool):
            raise ValueError("Only preregistered target weights .25/.5 are supported")
        self.base = base
        self.target_weight = target_weight

    def encode_descriptions(self, descriptions: Sequence[str]) -> torch.Tensor:
        full = self.base.encode_descriptions(descriptions)
        result = full.clone()
        for index, description in enumerate(descriptions):
            plan = parse_grounding_text(description)
            if plan is None:
                continue
            cue = self.base.encode_descriptions([plan.target_description])[0]
            combined = (1 - self.target_weight) * full[index] + self.target_weight * cue
            norm = combined.norm(p=2)
            if not torch.isfinite(combined).all() or not torch.isfinite(norm) or norm <= 0:
                raise ValueError("Invalid role-conditioned vector")
            result[index] = combined / norm
        return result
