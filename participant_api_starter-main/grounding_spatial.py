"""Conservative absolute image-position hints for one grounding description.

This parser leaves the model text intact. It recognizes only a short, explicit
position in the first target clause; ambiguous attachment returns no hint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AbsolutePosition:
    horizontal: int | None
    vertical: int | None

    def __post_init__(self) -> None:
        for value in (self.horizontal, self.vertical):
            if value is not None and (type(value) is not int or value not in (-1, 0, 1)):
                raise ValueError("Absolute position axes must be -1, 0, 1 or None")
        if self.horizontal is None and self.vertical is None:
            raise ValueError("Absolute position requires at least one axis")


_FIRST_CLAUSE = re.compile(r"[,，.;；。!?！？\r\n]")
_NEGATION = re.compile(
    r"\b(?:not|no|never|neither|nor|without|except|exclude|excluding|instead|rather|"
    r"isn't|isnt|aren't|arent|cannot|can't|cant)\b|"
    r"不(?:是|在|位于|处于|要|选|含|存在)|(?:并|而)非|"
    r"非(?=左|右|上|下|中央|中心|中间|位于|在|处于)|"
    r"未(?:在|位于|处于|见|发现|出现)|没有|无(?:一|任何|法|需|须)|排除|除外",
    re.IGNORECASE,
)
_REQUEST_PREFIX = re.compile(
    r"^(?:(?:please\s+)?(?:locate|find|identify|select|show|mark|box)\s+|"
    r"(?:请)?(?:框出|框选|定位|标出|找出|寻找|识别|选择)\s*)",
)
_EN_RELATION = re.compile(
    r"\b(?:near|nearby|beside|between|behind|above|below|with|without|adjacent|"
    r"around|along|toward|towards|facing|heading|pointing|opposite|"
    r"next\s+to|in\s+front\s+of|relative\s+to)\b",
)
_EN_OTHER_SUBJECT = re.compile(
    r"\b(?:and|or|but|while|whereas|which|whose|that|its|their|his|her|"
    r"has|have|contains|containing|features|featuring|includes|including)\b",
)
_ZH_RELATION = re.compile(
    r"附近|旁边|旁侧|旁|靠近|接近|紧邻|邻近|相邻|挨着|之间|周围|"
    r"相对|参照|对面|朝向|面向|向左|向右|左转|右转|以及|或者|或|且|而",
)


def _position_labels() -> dict[str, AbsolutePosition]:
    labels: dict[str, AbsolutePosition] = {}
    for vertical, vertical_words in ((-1, ("top", "upper")), (1, ("bottom", "lower"))):
        for horizontal, word in ((-1, "left"), (1, "right")):
            for vertical_word in vertical_words:
                labels[f"{vertical_word} {word}"] = AbsolutePosition(horizontal, vertical)
                labels[f"{word} {vertical_word}"] = AbsolutePosition(horizontal, vertical)
        for middle in ("center", "centre", "middle"):
            for vertical_word in vertical_words:
                labels[f"{vertical_word} {middle}"] = AbsolutePosition(0, vertical)
    for horizontal, word in ((-1, "left"), (1, "right")):
        for middle in ("center", "centre", "middle"):
            labels[f"{middle} {word}"] = AbsolutePosition(horizontal, 0)
            labels[f"{word} {middle}"] = AbsolutePosition(horizontal, 0)
        labels[word] = AbsolutePosition(horizontal, None)
    for word in ("top", "upper"):
        labels[word] = AbsolutePosition(None, -1)
    for word in ("bottom", "lower"):
        labels[word] = AbsolutePosition(None, 1)
    for word in ("center", "centre", "middle", "central"):
        labels[word] = AbsolutePosition(0, 0)
    return labels


_EN_POSITIONS = _position_labels()
_EN_POSITION = re.compile(
    r"\b(?:" + "|".join(re.escape(label) for label in sorted(_EN_POSITIONS, key=len, reverse=True)) + r")\b",
)
_FRAME = r"(?:image|picture|photo|photograph)"
_FRAME_WORDS = r"(?:(?:the|this|entire|whole)\s+)*"
_EN_FRAME_OF = re.compile(r"\bof\s+" + _FRAME_WORDS + _FRAME + r"\b")
_EN_LOCATION_SUFFIX = re.compile(
    r"^\s*(?:(?:corner|side|edge|half|region|area|part|portion|section|quadrant|zone)\b\s*)?"
    r"(?:of\s+" + _FRAME_WORDS + _FRAME + r"\b\s*)?",
)
_EN_LEADING = re.compile(
    r"^(?:(?:the|a|an)\s+)?$|^(?:in|at|on)\s+(?:the\s+)?$|"
    r"^(?:(?:the|this)\s+)?" + _FRAME + r"'s\s+$",
)
_EN_TARGET_LOCATION = re.compile(
    r"\b(?:in|at|on)\s+(?:(?:the|this)\s+)?(?:" + _FRAME + r"'s\s+)?$|"
    r"\b(?:is|lies|stands|sits|located|situated|positioned)\s+(?:the\s+)?$",
)
_EN_PART_NOUN = re.compile(
    r"^(?:angles?|hands?|wings?|decks?|banks?|turns?|turning|schools?|sized|floors?)\b",
)

_ZH_POSITIONS = {
    "左上": AbsolutePosition(-1, -1), "右上": AbsolutePosition(1, -1),
    "左下": AbsolutePosition(-1, 1), "右下": AbsolutePosition(1, 1),
    "左中": AbsolutePosition(-1, 0), "中左": AbsolutePosition(-1, 0),
    "右中": AbsolutePosition(1, 0), "中右": AbsolutePosition(1, 0),
    "上中": AbsolutePosition(0, -1), "中上": AbsolutePosition(0, -1),
    "下中": AbsolutePosition(0, 1), "中下": AbsolutePosition(0, 1),
    "顶部中央": AbsolutePosition(0, -1), "底部中央": AbsolutePosition(0, 1),
    "上方中间": AbsolutePosition(0, -1), "下方中间": AbsolutePosition(0, 1),
    "左侧中部": AbsolutePosition(-1, 0), "右侧中部": AbsolutePosition(1, 0),
    "正左": AbsolutePosition(-1, None), "正右": AbsolutePosition(1, None),
    "正上": AbsolutePosition(None, -1), "正下": AbsolutePosition(None, 1),
    "左": AbsolutePosition(-1, None), "右": AbsolutePosition(1, None),
    "上": AbsolutePosition(None, -1), "下": AbsolutePosition(None, 1),
    "正中央": AbsolutePosition(0, 0), "正中心": AbsolutePosition(0, 0),
    "正中间": AbsolutePosition(0, 0), "中央": AbsolutePosition(0, 0),
    "中心": AbsolutePosition(0, 0), "中间": AbsolutePosition(0, 0),
    "中部": AbsolutePosition(0, 0), "正中": AbsolutePosition(0, 0),
}
_ZH_POSITION = re.compile(
    "|".join(re.escape(label) for label in sorted(_ZH_POSITIONS, key=len, reverse=True) if len(label) > 1)
    + r"|[左右上下](?=角|方|部|侧|边|面|端|区|半|的|\s|$)",
)
_ZH_FRAME = r"(?:(?:整幅|整张|全幅|这张|这幅|该)?(?:图像|图片|照片|画面|图中|图)|全图)的?"
_ZH_FRAME_PATTERN = re.compile(_ZH_FRAME)
_ZH_LEADING = re.compile(r"^(?:在|位于|坐落于)?(?:" + _ZH_FRAME + r")?$")
_ZH_TARGET_LOCATION = re.compile(r"(?:在|位于|坐落于)(?:" + _ZH_FRAME + r")?$")
_ZH_LOCATION_SUFFIX = re.compile(r"^(?:角落|角|方|部|侧|边缘|边|面|端|半部|半边|区域|区|部分|位置|处)*")


def _english_position(clause: str) -> AbsolutePosition | None:
    if _EN_RELATION.search(clause) or _EN_OTHER_SUBJECT.search(clause):
        return None
    # Even an explicit image-frame suffix is insufficient when another `of`
    # binds the phrase to an object or one of its parts.
    if re.search(r"\bof\b", _EN_FRAME_OF.sub("", clause)):
        return None
    if re.search(r"\b(?!" + _FRAME + r"\b)\w+'s\b", clause):
        return None
    matches = list(_EN_POSITION.finditer(clause))
    if len(matches) != 1:
        return None
    match = matches[0]
    prefix = clause[:match.start()]
    suffix = _EN_LOCATION_SUFFIX.sub("", clause[match.end():], count=1).strip()
    if _EN_LEADING.fullmatch(prefix):
        if _EN_PART_NOUN.search(suffix):
            return None
        return _EN_POSITIONS[match.group()]
    locator = _EN_TARGET_LOCATION.search(prefix)
    if locator is None or not prefix[:locator.start()].strip() or suffix:
        return None
    return _EN_POSITIONS[match.group()]


def _chinese_position(clause: str) -> AbsolutePosition | None:
    if _ZH_RELATION.search(clause):
        return None
    # Mask frame words without changing offsets: the 中 in 图中 belongs to
    # the frame, so 图中下方 must not become the two-axis label 中下.
    position_text = _ZH_FRAME_PATTERN.sub(lambda match: " " * len(match.group()), clause)
    matches = list(_ZH_POSITION.finditer(position_text))
    if len(matches) != 1:
        return None
    match = matches[0]
    prefix = clause[:match.start()].strip()
    suffix = _ZH_LOCATION_SUFFIX.sub("", clause[match.end():], count=1).strip()
    if _ZH_LEADING.fullmatch(prefix):
        # One initial 的 connects location to target. Another 的 can instead
        # make the positioned noun a reference object: 右上角仓库的汽车.
        target = suffix.removeprefix("的")
        if "的" in target:
            return None
        return _ZH_POSITIONS[match.group()]
    locator = _ZH_TARGET_LOCATION.search(prefix)
    if locator is None or not prefix[:locator.start()].strip() or suffix:
        return None
    if re.search(r"有|拥有|包含|带有|具有|它的|其", prefix[:locator.start()]):
        return None
    return _ZH_POSITIONS[match.group()]


def parse_absolute_position(description: str) -> AbsolutePosition | None:
    """Return a clear target/frame hint, or None without rewriting any text.

    Additional clauses can describe reference objects. Relations, negation,
    multiple hints and uncertain noun attachment deliberately disable the hint.
    Cardinal geography and motion words are not treated as image coordinates.
    """
    if not isinstance(description, str) or not description.strip():
        return None
    clause = _FIRST_CLAUSE.split(description.strip(), maxsplit=1)[0].strip()
    clause = re.sub(r"\s+", " ", clause.casefold().replace("’", "'").replace("–", "-").replace("—", "-"))
    clause = re.sub(r"(?<=\w)-(?=\w)", " ", clause).strip()
    if not clause or _NEGATION.search(clause):
        return None
    clause = _REQUEST_PREFIX.sub("", clause, count=1).strip()
    if re.search(r"[\u3400-\u9fff]", clause):
        return _chinese_position(clause)
    return _english_position(clause)
