"""Conservative description parsing for the opt-in reference-first experiment.

Relations always describe the target relative to the reference. This module is
independent of the API and model runtimes. Unsupported or ambiguous expressions
return ``None`` so the caller can keep its complete-description inference path.
Only explicit grammar is recognized; object categories are never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceSpec:
    description: str
    relation: str


@dataclass(frozen=True)
class ReferencePlan:
    original_description: str
    target_description: str
    references: tuple[ReferenceSpec, ...]
    absolute_position: tuple[int | None, int | None] | None


_INVERSE = {
    "left_of": "right_of", "right_of": "left_of",
    "above": "below", "below": "above", "near": "near",
}
_EN_REL = (
    r"(?:to\s+the\s+left\s+of|on\s+the\s+left\s+of|left\s+of|"
    r"to\s+the\s+right\s+of|on\s+the\s+right\s+of|right\s+of|"
    r"above|below|next\s+to|beside|near)"
)
_EN_INLINE = re.compile(
    rf"^(?P<target>.+?)\s+(?:(?:that|which)\s+)?(?:is\s+)?"
    rf"(?P<relation>{_EN_REL})\s+(?P<reference>.+)$", re.I,
)
_EN_DIRECT = re.compile(
    rf"^(?:(?:it|the\s+target|this\s+target|that\s+target|target)"
    rf"\s+is\s+)?(?P<relation>{_EN_REL})\s+(?P<reference>.+)$", re.I,
)
_EN_REVERSE = re.compile(
    rf"^(?P<reference>.+?)\s+is\s+(?P<relation>{_EN_REL})\s+"
    r"(?:it|the\s+target|this\s+target|that\s+target|target)$", re.I,
)
_EN_REL_START = re.compile(
    rf"^(?:(?:it|the\s+target|this\s+target|that\s+target|target)"
    rf"\s+is\s+)?{_EN_REL}\s+", re.I,
)
_EN_REL_INSIDE = re.compile(rf"\b{_EN_REL}\b", re.I)
_ZH_REL = r"(?:左侧|左边|左方|右侧|右边|右方|上方|上侧|上面|下方|下侧|下面|旁边|附近)"
_ZH_NEAR = r"(?:靠近|临近|邻近|紧邻|邻接|毗邻)"
_ZH_INLINE = re.compile(
    rf"^(?P<target>.+?)(?:位于|处于|在)\s*(?P<reference>.+?)"
    rf"\s*(?:的\s*)?(?P<relation>{_ZH_REL})$",
)
_ZH_INLINE_NEAR = re.compile(
    rf"^(?P<target>.+?){_ZH_NEAR}\s*(?P<reference>.+)$",
)
_ZH_DIRECT = re.compile(
    rf"^(?:位于|处于|在)\s*(?P<reference>.+?)"
    rf"\s*(?:的\s*)?(?P<relation>{_ZH_REL})$",
)
_ZH_DIRECT_NEAR = re.compile(rf"^{_ZH_NEAR}\s*(?P<reference>.+)$")
_ZH_REVERSE = re.compile(
    rf"^(?P<reference>.+?)(?:位于|处于|在)\s*"
    rf"(?:它|该目标|这个目标|目标|该物体)(?:的\s*)?(?P<relation>{_ZH_REL})$",
)
_ZH_SIDE_HAS = re.compile(
    rf"^(?:其|它的|该目标的|这个目标的|目标的)?(?P<relation>{_ZH_REL})"
    r"(?:有|是)\s*(?P<reference>.+)$",
)
_NEGATIVE = re.compile(
    r"\b(?:no|not|never|neither|nor|without|except|excluding|unless|either|or|but)\b|"
    r"\b(?:isn|aren|wasn|weren|doesn|don|didn|shouldn|wouldn|won)['’]t\b|"
    r"没有|并非|不是|不(?:在|位于|处于|靠近|临近|紧邻|毗邻|是|要|应|能)|"
    r"既不|而非|未(?:在|位于|处于)|非(?=左|右|上|下|旁|附)|或者|或是|除外|除了",
    re.I,
)
_UNSUPPORTED = re.compile(
    r"\b(?:between|behind|inside|outside|within|around|opposite|across|among|amongst|"
    r"overlapping|overlaps|underneath|under|over|atop)\b|\bin\s+front\s+of\b|"
    r"\b(?:upper|lower|top|bottom)[\s-]+(?:left|right)\s+of\b|"
    r"之间|斜(?:上|下)|(?:左|右)(?:上|下)(?:方|侧)|"
    r"包围|穿过|交叉|重叠|前方|后方|内侧|外侧|内部|外部|里面|外面|围绕",
    re.I,
)
_EN_COLOR = (
    r"(?:red|orange|yellow|green|blue|cyan|purple|pink|white|gray|grey|black|brown|"
    r"beige|tan|silver|golden|gold|navy|teal|violet|maroon|turquoise|magenta|burgundy)"
)
_EN_COLOR_LIST = re.compile(rf"\b{_EN_COLOR}(?:\s+and\s+{_EN_COLOR})+\b", re.I)
_ZH_COLOR_LIST = re.compile(r"(?:红|橙|黄|绿|蓝|青|紫|粉|白|灰|黑|棕)色(?:和(?:红|橙|黄|绿|蓝|青|紫|粉|白|灰|黑|棕)色)+")
_EN_COPULA = re.compile(r"\b(?:is|are|was|were|has|have|whose|which|that)\b", re.I)
_ZH_NESTED = re.compile(
    rf"(?:位于|处于|在).+{_ZH_REL}|{_ZH_NEAR}|"
    rf".+{_ZH_REL}的.+",
)
_ZH_FRAME_POSITION = re.compile(
    rf"(?:(?:位于|处于|在)\s*)?(?:图像|图片|画面|影像|图中)(?:的)?\s*"
    rf"(?:左上角|右上角|左下角|右下角|左侧中央|右侧中央|上方中央|下方中央|正中间|中央|中心|中间|{_ZH_REL})",
)
_AMBIGUOUS_PRONOUN = re.compile(
    r"\b(?:it|its|they|them|their|his|her|self|itself|themselves|himself|herself|"
    r"myself|yourself|yourselves|ourselves)\b|它|自己|自身|本身|该(?:目标|物体|对象)", re.I,
)
_FRAME_NAMES = frozenset({"图像", "图片", "画面", "影像", "图中", "image", "picture", "scene", "the image", "the picture", "the scene"})
_PRONOUNS = frozenset({"it", "this", "that", "they", "them", "its", "itself", "target", "the target", "this target", "that target", "它", "它们", "这个", "那个", "目标", "该目标", "本目标", "该物体", "该对象", "自己", "自身", "本身"})
_CLAUSE_SEPARATORS = re.compile(r"[,，;；。\n、]+|\.\s+(?=[A-Za-z\u3400-\u9fff])")


def _english_relation(text: str) -> str:
    value = " ".join(text.lower().split())
    if value.endswith("left of"):
        return "left_of"
    if value.endswith("right of"):
        return "right_of"
    return value if value in {"above", "below"} else "near"


def _chinese_relation(text: str) -> str:
    if text.startswith("左"):
        return "left_of"
    if text.startswith("右"):
        return "right_of"
    if text.startswith("上"):
        return "above"
    if text.startswith("下"):
        return "below"
    return "near"


def _split_english_and(clause: str) -> list[str]:
    """Split relation conjunctions while retaining adjective color conjunctions."""
    cuts = []
    color_spans = [match.span() for match in _EN_COLOR_LIST.finditer(clause)]
    for match in re.finditer(r"\s+and\s+", clause, re.I):
        if any(start <= match.start() < end for start, end in color_spans):
            continue
        suffix = clause[match.end():].strip()
        if _EN_REL_START.match(suffix) or _EN_REVERSE.fullmatch(suffix):
            cuts.append((match.start(), match.end()))
    parts, start = [], 0
    for left, right in cuts:
        parts.append(clause[start:left].strip())
        start = right
    parts.append(clause[start:].strip())
    return parts


def _clauses(text: str, *, english: bool) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_SEPARATORS.split(text.strip().rstrip(".。"))]
    if any(not part for part in parts):
        return []
    if english:
        return [item for part in parts for item in _split_english_and(part)]
    return parts


def _atomic_phrase(text: str) -> bool:
    if not text or text.lower() in _PRONOUNS or not re.search(r"[A-Za-z0-9\u3400-\u9fff]", text):
        return False
    if (_EN_REL_INSIDE.search(text) or _EN_COPULA.search(text)
            or _ZH_NESTED.search(_ZH_FRAME_POSITION.sub("", text))
            or _AMBIGUOUS_PRONOUN.search(text)):
        return False
    no_colors = _ZH_COLOR_LIST.sub("颜色", _EN_COLOR_LIST.sub("color", text))
    # A general object conjunction cannot safely become one reference identity.
    return not re.search(r"\b(?:and|or)\b|&|和|与|以及|及", no_colors, re.I)


_ABS_POSITIONS: dict[str, tuple[int | None, int | None]] = {
    "top left": (-1, -1), "upper left": (-1, -1),
    "top right": (1, -1), "upper right": (1, -1),
    "bottom left": (-1, 1), "lower left": (-1, 1),
    "bottom right": (1, 1), "lower right": (1, 1),
    "middle left": (-1, 0), "center left": (-1, 0), "left center": (-1, 0),
    "middle right": (1, 0), "center right": (1, 0), "right center": (1, 0),
    "top center": (0, -1), "upper center": (0, -1), "top middle": (0, -1),
    "bottom center": (0, 1), "lower center": (0, 1), "bottom middle": (0, 1),
    "left": (-1, None), "right": (1, None),
    "top": (None, -1), "upper": (None, -1),
    "bottom": (None, 1), "lower": (None, 1),
    "center": (0, 0), "middle": (0, 0),
    "左上角": (-1, -1), "右上角": (1, -1),
    "左下角": (-1, 1), "右下角": (1, 1),
    "左侧中央": (-1, 0), "右侧中央": (1, 0),
    "上方中央": (0, -1), "下方中央": (0, 1),
    "中央": (0, 0), "中心": (0, 0), "中间": (0, 0), "正中间": (0, 0),
    "左侧": (-1, None), "右侧": (1, None),
    "上方": (None, -1), "下方": (None, 1),
}
_EN_ABS_TOKEN = "|".join(
    re.escape(key).replace(r"\ ", r"[\s-]+")
    for key in sorted((key for key in _ABS_POSITIONS if key.isascii()), key=len, reverse=True)
)
_EN_ABS_PREFIX = re.compile(
    rf"^(?:(?:the|a|an)\s+)?(?P<position>{_EN_ABS_TOKEN})\b", re.I,
)
_EN_ABS_SUFFIX = re.compile(
    rf"\b(?:in|at|on)\s+(?:the\s+)?(?P<position>{_EN_ABS_TOKEN})"
    r"(?:\s+(?:corner|part|area|side|portion|region))?"
    r"(?:\s+of\s+(?:the\s+)?(?:image|picture|scene))?$", re.I,
)
_ZH_ABS_TOKEN = "|".join(sorted((key for key in _ABS_POSITIONS if not key.isascii()), key=len, reverse=True))
_ZH_ABS_PREFIX = re.compile(
    rf"^(?:(?:位于|在)\s*)?(?:(?:图像|图片|画面|影像|图中)(?:的)?\s*)?"
    rf"(?P<position>{_ZH_ABS_TOKEN})(?:的)?(?=.+)",
)
_ZH_ABS_SUFFIX = re.compile(
    rf"(?:位于|在)\s*(?:(?:图像|图片|画面|影像|图中)(?:的)?\s*)?"
    rf"(?P<position>{_ZH_ABS_TOKEN})$",
)


def _absolute_position(target: str) -> tuple[bool, tuple[int | None, int | None] | None]:
    hints = []
    if not re.search(r"\b(?:with|having|containing|whose)\b", target, re.I):
        for pattern in (_EN_ABS_PREFIX, _EN_ABS_SUFFIX, _ZH_ABS_PREFIX, _ZH_ABS_SUFFIX):
            match = pattern.search(target)
            if match:
                key = " ".join(match.group("position").lower().replace("-", " ").split())
                # Simple adjective prefixes such as "left bank" are not clear
                # image-frame positions. Composite corners/centers are explicit.
                if pattern is _EN_ABS_PREFIX and " " not in key:
                    continue
                hints.append(_ABS_POSITIONS[key])
    if hints and any(hint != hints[0] for hint in hints[1:]):
        return False, None
    return True, hints[0] if hints else None


def _parse_english(text: str) -> tuple[str, list[ReferenceSpec]] | None:
    clauses = _clauses(text, english=True)
    if not clauses:
        return None
    refs = []
    inline = _EN_INLINE.fullmatch(clauses[0])
    if inline:
        target = inline.group("target").strip()
        refs.append(ReferenceSpec(inline.group("reference").strip(), _english_relation(inline.group("relation"))))
    else:
        target = clauses[0]
    for clause in clauses[1:]:
        reverse = _EN_REVERSE.fullmatch(clause)
        direct = _EN_DIRECT.fullmatch(clause) if reverse is None else None
        match = reverse or direct
        if match is None:
            return None
        relation = _english_relation(match.group("relation"))
        if reverse:
            relation = _INVERSE[relation]
        refs.append(ReferenceSpec(match.group("reference").strip(), relation))
    return target, refs


def _parse_chinese(text: str) -> tuple[str, list[ReferenceSpec]] | None:
    clauses = _clauses(text, english=False)
    if not clauses:
        return None
    refs = []
    inline = _ZH_INLINE.fullmatch(clauses[0])
    near = _ZH_INLINE_NEAR.fullmatch(clauses[0]) if inline is None else None
    first = inline or near
    if first and first.group("reference").strip().lower() not in _FRAME_NAMES:
        target = first.group("target").strip()
        relation = _chinese_relation(first.group("relation")) if inline else "near"
        refs.append(ReferenceSpec(first.group("reference").strip(), relation))
    else:
        target = clauses[0]
    for clause in clauses[1:]:
        reverse = _ZH_REVERSE.fullmatch(clause)
        side_has = _ZH_SIDE_HAS.fullmatch(clause) if reverse is None else None
        direct = _ZH_DIRECT.fullmatch(clause) if reverse is None and side_has is None else None
        near = _ZH_DIRECT_NEAR.fullmatch(clause) if reverse is None and side_has is None and direct is None else None
        match = reverse or side_has or direct or near
        if match is None:
            return None
        relation = _chinese_relation(match.group("relation")) if match is not near else "near"
        if reverse or side_has:
            relation = _INVERSE[relation]
        refs.append(ReferenceSpec(match.group("reference").strip(), relation))
    return target, refs


def parse_reference_plan(description: str) -> ReferencePlan | None:
    """Parse one target and 1--3 explicit anchors; otherwise decline the trial.

    Supported examples include ``lawn to the left of a road and below a
    building``, ``lawn, road is to the right of it``, ``草坪，在道路左侧`` and
    ``草坪，右侧有道路``. Reference-side position attributes remain in the
    reference phrase and never supply the target's absolute-position hint.
    """
    if not isinstance(description, str) or not description.strip():
        return None
    if _NEGATIVE.search(description) or _UNSUPPORTED.search(description):
        return None
    candidates = []
    for parse in (_parse_english, _parse_chinese):
        parsed = parse(description)
        if parsed is None:
            continue
        target, refs = parsed
        if not 1 <= len(refs) <= 3 or not _atomic_phrase(target):
            continue
        if any(not _atomic_phrase(ref.description) or ref.description.lower() in _FRAME_NAMES
               for ref in refs):
            continue
        normalized_target = " ".join(target.lower().split())
        normalized_refs = [" ".join(ref.description.lower().split()) for ref in refs]
        if normalized_target in normalized_refs or len(set(normalized_refs)) != len(refs):
            continue
        valid_position, position = _absolute_position(target)
        if not valid_position:
            continue
        candidates.append(ReferencePlan(description, target, tuple(refs), position))
    if not candidates or any(plan != candidates[0] for plan in candidates[1:]):
        return None
    return candidates[0]
