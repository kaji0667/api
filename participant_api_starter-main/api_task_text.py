"""Task-side input adaptation; the router and adapter preserve the source question.

Remove explicit protocol packaging without summarizing or translating grounding
descriptions. Counting extracts a target category; targets never come from
choices, identifiers or annotations.
"""

from __future__ import annotations

import re


class TaskInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_ANSWER_INSTRUCTION = re.compile(
    r"(?:^|(?<=[.!?。！？])\s+|\n\s*)"
    r"(?:only\s+(?:answer|output|return)|(?:please\s+)?(?:answer|output|return)\s+"
    r"(?:only|a\s+(?:pixel\s+)?(?:box|bounding\s+box)|the\s+(?:answer|coordinates))|"
    r"your\s+answer\s+must\s+be|只(?:需|能)?(?:回答|输出|返回))\b.*$",
    re.IGNORECASE | re.DOTALL,
)

# Grounding-only format directives. A sentence/line boundary and an explicit
# box/coordinate noun are required: visual descriptions such as "a box-shaped
# building" or a relative clause containing "return" are not instructions.
_BBOX_ANSWER_INSTRUCTION = re.compile(
    r"(?:^|(?<=[.!?。！？])\s+|\n\s*)"
    r"(?:(?:please\s+)?(?:return|output|provide|give|report)\s+"
    r"(?:(?:only|just)\s+)?(?:(?:a|an|the|its)\s+)?"
    r"(?:(?:pixel|axis[- ]aligned)\s+)?"
    r"(?:bbox\b|bounding\s+box\b|box\s+coordinates\b|coordinates\b)|"
    r"(?:your\s+)?(?:response|output)\s+(?:must|should)\s+be\s+"
    r"(?:(?:a|the)\s+)?(?:bbox\b|bounding\s+box\b|coordinates\b)|"
    r"(?:请)?(?:只(?:需|能)?(?:回答|输出|返回)|返回|输出|给出)\s*"
    r"(?:该目标的?|目标的?|物体的?)?(?:边界框|边框|坐标|bbox\b))"
    r".*$", re.IGNORECASE | re.DOTALL,
)
_DESCRIPTION_MARKER = re.compile(
    r"\bdescription\s*[:：]|(?:目标描述|物体描述|描述)\s*[:：]", re.IGNORECASE,
)


def _source(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise TaskInputError("empty_task_text", "Task requires nonempty source text")
    return question.strip()


def _without_answer_instruction(text: str) -> str:
    return _ANSWER_INSTRUCTION.sub("", text).strip()


def grounding_text(question: str) -> str:
    """Keep the entire explicit Description payload, including relationships."""
    text = _source(question)
    markers = list(_DESCRIPTION_MARKER.finditer(text))
    if len(markers) > 1:
        raise TaskInputError("ambiguous_description", "Multiple description payloads")
    if markers:
        text = text[markers[0].end():]
    # Strip bbox formatting before the shared, narrower answer wrapper. This
    # preserves every target/reference sentence and interior newline verbatim.
    text = _BBOX_ANSWER_INSTRUCTION.sub("", text).strip()
    text = _without_answer_instruction(text)
    if not text:
        raise TaskInputError("empty_task_text", "Missing object description")
    return text


def detection_text(question: str) -> str:
    """Convert explicit existence-question wrapping to the intact object phrase."""
    text = grounding_text(question)
    presence = re.fullmatch(
        r"(?:is\s+there\s+(?:an?\s+)?|are\s+there\s+(?:any\s+)?|"
        r"does\s+(?:the|this)\s+(?:image|picture)\s+(?:contain|show)\s+(?:an?\s+)?)"
        r"(.+?)(?:\s+(?:visible\s+)?in\s+(?:the|this)\s+(?:image|picture))?[?.!]*",
        text, re.IGNORECASE | re.DOTALL,
    )
    if presence:
        return presence.group(1).strip()
    return text


# Known inflections only, not a target whitelist. Unknown bare class labels are
# retained because the original VOC reader and counter accept arbitrary labels.
# The model's class aliases/template are applied by the original count encoder.
_CLASSES = (
    "airplane", "airport", "baseball diamond", "basketball court", "bridge",
    "chimney", "dam", "expressway service area", "expressway toll station",
    "golf field", "golffield", "ground track field", "harbor", "harbour",
    "overpass", "ship", "stadium", "storage tank", "tennis court",
    "train station", "vehicle", "windmill", "wind turbine", "car", "boat",
    "building", "house", "helipad", "soccer ball field", "soccer field",
    "swimming pool", "helicopter", "roundabout", "small vehicle", "large vehicle",
    "container crane", "plane", "oil tank",
    "truck", "golf course", "trainstation",
)
_CLASS_FORMS = {name: name for name in _CLASSES}
_CLASS_FORMS.update({name + "s": name for name in _CLASSES})
_CLASS_FORMS.update({"houses": "house"})
# Recover the original VOC class name from equivalent question vocabulary.
# This remains task-side normalization, not a whitelist or answer lookup.
_CLASS_FORMS.update({
    "plane": "airplane", "planes": "airplane",
    "aeroplane": "airplane", "aeroplanes": "airplane",
})
_UNSUPPORTED_COUNT_CONDITION = re.compile(
    r"\b(?:near|beside|between|except|excluding|exclude|within|outside|inside|"
    r"above|below|left|right|top|bottom|upper|lower|closest|nearest|farthest|"
    r"red|green|blue|white|black|gray|grey|yellow|brown|colored|colour|color|"
    r"and|or|not|only|are|is|there|image|picture|how|many|"
    r"on|in|at|along|around|with|without|under|over|that|which|whose|than)\b",
    re.IGNORECASE,
)
_COUNT_SUFFIX = re.compile(
    r"\s+\(?\b(?:are|is|were|was|there|can|do|does|have|has|appear|appears|"
    r"visible|present|shown|seen|parked|moving|moored|anchored|"
    r"located|situated|near|beside|between|except|excluding|exclude|within|"
    r"outside|inside|above|below|left|right|top|bottom|upper|lower|closest|"
    r"nearest|farthest|on|in|at|along|around|with|without|under|over|"
    r"that|which|whose)\b",
    re.IGNORECASE,
)
_COUNT_MODIFIER = re.compile(
    r"^(?:red|green|blue|white|black|gray|grey|yellow|brown|colored|"
    r"coloured|parked|moving|circular|rectangular|small|large)\s+", re.IGNORECASE,
)


def counting_text(question: str) -> str:
    """Extract one target category for the original class-conditioned counter."""
    text = " ".join(_without_answer_instruction(_source(question)).split())
    if len(re.findall(r"\bhow\s+many\b", text, re.IGNORECASE)) > 1:
        raise TaskInputError("unsupported_count_query", "Counting requires one class query")
    start = re.search(r"\bhow\s+many\s+", text, re.IGNORECASE)
    if start:
        text = text[start.start():]
    else:
        wrapper = re.match(
            r"(?:(?:what\s+(?:is|are)\s+)?(?:the\s+)?(?:total\s+)?number\s+of\s+|"
            r"count\s+(?:(?:the\s+)?(?:total\s+)?number\s+of\s+)?)",
            text, re.IGNORECASE,
        )
        if wrapper:
            text = "How many " + text[wrapper.end():]
    query = re.fullmatch(
        r"how\s+many\s+(.+?)(?:\s+(?:"
        r"(?:are\s+)?(?:there\s+)?(?:visible|present|shown|seen)(?:\s+in\s+(?:the|this)\s+(?:image|picture))?|"
        r"(?:are\s+(?:there\s+)?|can\s+be\s+seen\s+|do\s+you\s+see\s+)?"
        r"in\s+(?:the|this)\s+(?:image|picture)|are\s+there|do\s+you\s+see))?[?.!]*",
        text, re.IGNORECASE,
    )
    target = query.group(1).strip() if query else text
    target = " ".join(target.casefold().replace("_", " ").replace("-", " ").split())
    target = target.rstrip("?.!").strip()
    suffix = _COUNT_SUFFIX.search(target)
    if suffix:
        target = target[:suffix.start()].strip().rstrip(",").strip()
    target = re.sub(r"^of\s+(?:the\s+)?", "", target)
    target = re.sub(r"^(?:(?:the|these|those|all|any|total)\s+)+", "", target)
    uncolored = re.sub(
        r"\b(?:red|green|blue|white|black|gray|grey|yellow|brown|colored|coloured)\s+",
        "", target,
    )
    if uncolored != target and uncolored in _CLASS_FORMS:
        target = uncolored
    # Preserve genuine composite labels such as small vehicle / large vehicle.
    while target not in _CLASS_FORMS and _COUNT_MODIFIER.match(target):
        target = _COUNT_MODIFIER.sub("", target, count=1)
    if (not target or len(target) > 100 or re.search(r"[?!,;:]", target)
            or _UNSUPPORTED_COUNT_CONDITION.search(target)):
        raise TaskInputError("unsupported_count_query", "Counting requires an identifiable single class query")
    if target in {"object", "objects", "thing", "things", "anything", "everything"}:
        raise TaskInputError("unsupported_count_query", "Counting requires a specific class query")
    return _CLASS_FORMS.get(target, target)


def count_option_matches(value: str, count: int) -> bool:
    """Interpret explicit numeric options without rounding or clamping a count."""
    text = value.strip().casefold()
    unit = r"(?:\s*(?:个|架|艘|辆|座|处|只|objects?|ships?|airplanes?|vehicles?))?"
    exact = re.fullmatch(r"(\d+)" + unit, text)
    if exact:
        return count == int(exact.group(1))
    interval = re.fullmatch(r"(?:between\s+)?(\d+)\s*(?:-|–|—|to|and|至|到)\s*(\d+)" + unit, text)
    if interval:
        lo, hi = map(int, interval.groups())
        return lo <= count <= hi if lo <= hi else False
    bound = re.fullmatch(r"(>=|≥|>|<=|≤|<)\s*(\d+)" + unit, text)
    if bound:
        op, n = bound.group(1), int(bound.group(2))
        return {">=": count >= n, "≥": count >= n, ">": count > n,
                "<=": count <= n, "≤": count <= n, "<": count < n}[op]
    bound = re.fullmatch(r"(at least|more than|over|at most|less than|under)\s+(\d+)" + unit, text)
    if bound:
        op, n = bound.group(1), int(bound.group(2))
        return {"at least": count >= n, "more than": count > n, "over": count > n,
                "at most": count <= n, "less than": count < n, "under": count < n}[op]
    bound = re.fullmatch(r"(\d+)\s*(\+|or more|or fewer)" + unit, text)
    if bound:
        n, op = int(bound.group(1)), bound.group(2)
        return count <= n if op == "or fewer" else count >= n
    return False
