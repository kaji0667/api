"""Web adapter for single-image text-guided semantic segmentation."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from ..config import TaskConfig
from .common import decode_data_url, png_data_url, release_model_resources, resolve_checkpoint


PredictorFactory = Callable[..., Any]


_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
_CATEGORY_TRANSLATIONS = (
    ("高速公路收费站", "expressway toll station"),
    ("高速公路服务区", "expressway service area"),
    ("风力发电机", "windmill"),
    ("铁路火车站", "train station"),
    ("地面田径场", "ground track field"),
    ("操场", "ground track field"),
    ("棒球场", "baseball field"),
    ("篮球场", "basketball court"),
    ("高尔夫球场", "golf field"),
    ("田径场", "ground track field"),
    ("网球场", "tennis court"),
    ("火车站", "train station"),
    ("储油罐", "storage tank"),
    ("储罐", "storage tank"),
    ("立交桥", "overpass"),
    ("体育场", "stadium"),
    ("飞机场", "airport"),
    ("机场", "airport"),
    ("飞机", "airplane"),
    ("桥梁", "bridge"),
    ("大桥", "bridge"),
    ("桥", "bridge"),
    ("烟囱", "chimney"),
    ("大坝", "dam"),
    ("水坝", "dam"),
    ("港口", "harbor"),
    ("港湾", "harbor"),
    ("轮船", "ship"),
    ("船舶", "ship"),
    ("船", "ship"),
    ("车辆", "vehicle"),
    ("汽车", "vehicle"),
    ("卡车", "vehicle"),
    ("风车", "windmill"),
)
_POSITION_TRANSLATIONS = (
    ("最左上方", "at the top left", "suffix"),
    ("最右上方", "at the top right", "suffix"),
    ("最左下方", "at the bottom left", "suffix"),
    ("最右下方", "at the bottom right", "suffix"),
    ("最上方", "topmost", "prefix"),
    ("最上面", "topmost", "prefix"),
    ("最顶部", "topmost", "prefix"),
    ("最下方", "bottommost", "prefix"),
    ("最下面", "bottommost", "prefix"),
    ("最底部", "bottommost", "prefix"),
    ("最左侧", "leftmost", "prefix"),
    ("最左边", "leftmost", "prefix"),
    ("最右侧", "rightmost", "prefix"),
    ("最右边", "rightmost", "prefix"),
    ("左上方", "at the top left", "suffix"),
    ("左上角", "at the top left", "suffix"),
    ("右上方", "at the top right", "suffix"),
    ("右上角", "at the top right", "suffix"),
    ("左下方", "at the bottom left", "suffix"),
    ("左下角", "at the bottom left", "suffix"),
    ("右下方", "at the bottom right", "suffix"),
    ("右下角", "at the bottom right", "suffix"),
    ("上方", "at the top", "suffix"),
    ("上面", "at the top", "suffix"),
    ("顶部", "at the top", "suffix"),
    ("下方", "at the bottom", "suffix"),
    ("下面", "at the bottom", "suffix"),
    ("底部", "at the bottom", "suffix"),
    ("左侧", "on the left", "suffix"),
    ("左边", "on the left", "suffix"),
    ("右侧", "on the right", "suffix"),
    ("右边", "on the right", "suffix"),
    ("中间", "in the center", "suffix"),
    ("中央", "in the center", "suffix"),
    ("中心", "in the center", "suffix"),
)
_SIZE_TRANSLATIONS = (
    ("最大的", "largest"),
    ("最小的", "smallest"),
    ("大型", "large"),
    ("较大", "large"),
    ("大的", "large"),
    ("小型", "small"),
    ("较小", "small"),
    ("小的", "small"),
    ("大", "large"),
    ("小", "small"),
)
_COLOR_TRANSLATIONS = (
    ("灰色", "gray"),
    ("白色", "white"),
    ("黑色", "black"),
    ("红色", "red"),
    ("蓝色", "blue"),
    ("绿色", "green"),
    ("黄色", "yellow"),
    ("棕色", "brown"),
    ("橙色", "orange"),
)
_IGNORED_CHINESE_PHRASES = (
    "遥感图像中",
    "遥感图中",
    "图像中",
    "图片中",
    "图中",
    "请帮我分割出",
    "请帮我提取",
    "请帮我找到",
    "请分割出",
    "请提取",
    "请找到",
    "分割出",
    "提取",
    "找到",
    "位于",
    "一个",
    "一架",
    "一艘",
    "一辆",
    "一座",
    "一处",
    "这个",
    "那个",
    "目标区域",
    "目标",
    "区域",
    "物体",
    "的",
    "在",
)


def _extract_translation(
    text: str,
    choices: tuple[tuple[str, str], ...],
) -> tuple[str, str]:
    for chinese, english in choices:
        if chinese in text:
            return text.replace(chinese, "", 1), english
    return text, ""


def translate_refseg_prompt(text: str) -> tuple[str, bool]:
    """Translate supported Chinese RefSeg phrases into the checkpoint's English prompt domain."""

    prompt = str(text).strip()
    if not prompt or not _CJK_PATTERN.search(prompt):
        return prompt, False

    working = re.sub(r"[，。！？、；：,.!?;:]", "", prompt)
    category = ""
    for chinese, english in _CATEGORY_TRANSLATIONS:
        if chinese in working:
            working = working.replace(chinese, "", 1)
            category = english
            break
    if category and any(chinese in working for chinese, _ in _CATEGORY_TRANSLATIONS):
        raise ValueError("当前中文转译只支持描述一个目标类别；包含关系的复杂描述请使用英文。")

    position = ""
    position_kind = ""
    for chinese, english, kind in _POSITION_TRANSLATIONS:
        if chinese in working:
            working = working.replace(chinese, "", 1)
            position = english
            position_kind = kind
            break

    working, size = _extract_translation(working, _SIZE_TRANSLATIONS)
    working, color = _extract_translation(working, _COLOR_TRANSLATIONS)
    for phrase in _IGNORED_CHINESE_PHRASES:
        working = working.replace(phrase, "")
    working = " ".join(working.split()).strip()

    if not category and re.fullmatch(r"[A-Za-z][A-Za-z -]*", working):
        category = working.lower()
        working = ""
    if not category:
        raise ValueError("暂时无法识别中文描述中的目标类别，请改用英文或使用已支持的遥感类别。")
    if working:
        raise ValueError("中文描述中包含暂不支持的关系词，请改用更简短的描述或直接输入英文。")

    descriptors = [item for item in (size, color) if item]
    if position_kind == "prefix":
        descriptors.insert(0, position)
    translated = " ".join(["the", *descriptors, category]).strip()
    if position_kind == "suffix":
        translated = f"{translated} {position}"
    return translated, True


def _default_predictor_factory(**kwargs: Any) -> Any:
    from tasks.refseg.inference import RefSegPredictor

    return RefSegPredictor(**kwargs)


class RefSegAdapter:
    """Translate the browser JSON contract into the task-owned predictor API."""

    def __init__(
        self,
        config: TaskConfig,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        self.config = config
        self.checkpoint = resolve_checkpoint(config, "HFSA_REFSEG_CHECKPOINT")
        self.device = os.environ.get("HFSA_REFSEG_DEVICE", "auto").strip() or "auto"
        self._predictor_factory = predictor_factory or _default_predictor_factory
        self._predictor: Any | None = None
        self._lock = threading.Lock()

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"RefSeg checkpoint not found: {self.checkpoint}")
            self._predictor = self._predictor_factory(
                checkpoint=self.checkpoint,
                device=self.device,
            )
        return self._predictor

    def predict(self, image: Any, text: str) -> dict[str, Any]:
        prompt = str(text).strip()
        if not prompt:
            raise ValueError("语义分割需要填写目标描述。")
        model_prompt, prompt_translated = translate_refseg_prompt(prompt)
        decoded_image = decode_data_url(image)

        with self._lock:
            prediction = self._get_predictor().predict(decoded_image, model_prompt)

        mask = np.asarray(prediction.mask, dtype=bool)
        probability = np.asarray(prediction.probability, dtype=np.float32)
        overlay = np.asarray(prediction.overlay, dtype=np.uint8)
        if mask.ndim != 2 or probability.shape != mask.shape:
            raise RuntimeError("RefSeg prediction mask and probability shapes do not match.")
        if overlay.shape != (*mask.shape, 3):
            raise RuntimeError("RefSeg overlay must match the mask size and contain three RGB channels.")

        summary = dict(prediction.summary())
        summary.update(
            {
                "input_prompt": prompt,
                "model_prompt": model_prompt,
                "prompt_translated": prompt_translated,
                "found_target": bool(mask.any()),
            }
        )
        return {
            "summary": summary,
            "images": {
                "overlay": png_data_url(overlay, "RGB"),
                "mask": png_data_url(mask.astype(np.uint8) * 255, "L"),
                "probability": png_data_url(
                    np.clip(probability * 255.0, 0, 255).astype(np.uint8),
                    "L",
                ),
            },
        }

    def close(self) -> None:
        predictor, self._predictor = self._predictor, None
        release_model_resources(predictor)
