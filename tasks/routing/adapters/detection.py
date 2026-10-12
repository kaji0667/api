"""Web adapter for the senior teammate's text-guided target detector."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from ..config import TaskConfig
from .common import decode_data_url, png_data_url, release_model_resources, resolve_checkpoint


PredictorFactory = Callable[..., Any]


def _default_predictor_factory(**kwargs: Any) -> Any:
    from tasks.detection.inference import TextGuidedDetectionPredictor

    return TextGuidedDetectionPredictor(**kwargs)


class DetectionAdapter:
    """Translate browser input into prompted detections and an annotated PNG."""

    def __init__(self, config: TaskConfig, predictor_factory: PredictorFactory | None = None) -> None:
        self.config = config
        self.checkpoint = resolve_checkpoint(config, "HFSA_DETECTION_CHECKPOINT")
        self.device = os.environ.get("HFSA_DETECTION_DEVICE", "auto").strip() or "auto"
        defaults = config.runtime_defaults
        self.imgsz = int(defaults.get("imgsz", 800))
        self.conf_thres = float(defaults.get("conf_thres", 0.15))
        self.iou_thres = float(defaults.get("iou_thres", 0.5))
        self.max_det = int(defaults.get("max_det", 300))
        self._predictor_factory = predictor_factory or _default_predictor_factory
        self._predictor: Any | None = None
        self._lock = threading.Lock()

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"Detection checkpoint not found: {self.checkpoint}")
            self._predictor = self._predictor_factory(
                checkpoint=self.checkpoint,
                device=self.device,
                imgsz=self.imgsz,
            )
        return self._predictor

    def predict(self, image: Any, text: str) -> dict[str, Any]:
        prompt = str(text).strip()
        if not prompt:
            raise ValueError("目标检测需要填写目标描述。")
        decoded_image = decode_data_url(image)
        with self._lock:
            prediction = self._get_predictor().predict(
                decoded_image,
                prompt,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                max_det=self.max_det,
            )
        boxes = np.asarray(prediction.boxes, dtype=np.float32)
        scores = np.asarray(prediction.scores, dtype=np.float32)
        visualization = np.asarray(prediction.visualization, dtype=np.uint8)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,) or scores.shape != (len(boxes),):
            raise RuntimeError("Detection boxes and scores have incompatible shapes.")
        if visualization.ndim != 3 or visualization.shape[2] != 3:
            raise RuntimeError("Detection visualization must be an RGB image.")
        return {
            "summary": {
                "prompt": str(prediction.text),
                "detection_count": int(len(boxes)),
                "latency_ms": float(prediction.latency_ms),
                "original_size": list(prediction.original_size),
                "conf_thres": self.conf_thres,
                "iou_thres": self.iou_thres,
            },
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "images": {"visualization": png_data_url(visualization, "RGB")},
        }

    def close(self) -> None:
        predictor, self._predictor = self._predictor, None
        release_model_resources(predictor)
