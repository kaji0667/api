"""API counting through the original positive-query tensor evaluator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from api_task_text import counting_text


def _load_model(checkpoint: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(checkpoint)).model


def _make_counter(model: Any, encoder: Any, device: torch.device) -> Any:
    # This is the same tensor-level counter called by tasks/counting/test.py.
    # No CountingPredictor or Web single-image Predictor is constructed.
    from tasks.counting.inference import ObjectCounter

    return ObjectCounter(model, encoder, device)


def _make_preprocessor() -> Any:
    from tasks.counting.inference import CountingImagePreprocessor

    return CountingImagePreprocessor(800)


class _CountingEncoder:
    """Adapt shared native CLIP to ObjectCounter's [B, token, feature] contract."""

    def __init__(self, encoder: Any, device: torch.device) -> None:
        self.encoder = encoder
        self.device = device

    def encode(self, target_class: str) -> torch.Tensor:
        features = self.encoder.encode_counting(target_class)
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != (1, 768):
            raise ValueError("Counting text encoder must return one feature vector")
        return features.to(self.device).unsqueeze(1)


@dataclass(frozen=True)
class NativeCountingPrediction:
    target_class: str
    count: int
    boxes: np.ndarray
    scores: np.ndarray
    latency_ms: float
    original_size: tuple[int, int]


class NativeCounting:
    """Run the original class-conditioned counter and count NMS detections."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "auto",
        encoder: Any = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Counting checkpoint is missing: {self.checkpoint}")
        requested = str(device).strip().lower()
        if requested in {"", "auto"}:
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        if encoder is None:
            from native_detection import NativeTextEncoder

            encoder = NativeTextEncoder(device=self.device)
        self.encoder = encoder
        self.counter = _make_counter(_load_model(self.checkpoint), _CountingEncoder(encoder, self.device), self.device)
        self.preprocessor = _make_preprocessor()

    @staticmethod
    def _to_bgr(image: Image.Image | str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Counting expects a BGR image shaped [H, W, 3]")
            return np.ascontiguousarray(image)
        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError("Counting image cannot be decoded")
            return bgr
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def predict(self, image: Image.Image | str | Path | np.ndarray, original_question: str) -> NativeCountingPrediction:
        target_class = counting_text(original_question)
        bgr = self._to_bgr(image)
        tensor, _, _ = self.preprocessor(bgr)
        started = time.perf_counter()
        result = self.counter.count(tensor, target_class, conf_thres=0.15, iou_thres=0.50, max_det=300)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        boxes = result["pred_boxes"].detach().cpu().clone()
        scores = result["pred_scores"].detach().cpu()
        if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1 or len(boxes) != len(scores):
            raise ValueError("Counting output has invalid detection shapes")
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise ValueError("Counting output contains non-finite detections")
        count = int(result["pred_count"])
        if count != len(boxes) or count < 0:
            raise ValueError("Counting result differs from its NMS detection count")
        if len(boxes):
            from ultralytics.utils.ops import scale_boxes

            scale_boxes(tensor.shape[2:], boxes, bgr.shape[:2])
        return NativeCountingPrediction(
            target_class=target_class,
            count=count,
            boxes=boxes.numpy().astype(np.float32, copy=False),
            scores=scores.numpy().astype(np.float32, copy=False),
            latency_ms=float(latency_ms),
            original_size=(int(bgr.shape[0]), int(bgr.shape[1])),
        )
