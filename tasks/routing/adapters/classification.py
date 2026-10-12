"""Web adapter for checkpoint-owning scene-classification inference."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

from ..config import TaskConfig
from .common import decode_data_url, release_model_resources, resolve_checkpoint, resolve_project_path


PredictorFactory = Callable[..., Any]


def _default_predictor_factory(**kwargs: Any) -> Any:
    from tasks.classification.inference import SceneClassificationPredictor

    return SceneClassificationPredictor(**kwargs)


class SceneClassificationAdapter:
    """Translate the browser image contract into scene-classification Top-K output."""

    def __init__(
        self,
        config: TaskConfig,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        self.config = config
        self.checkpoint = resolve_checkpoint(config, "HFSA_CLASSIFICATION_CHECKPOINT")
        pretrained = config.runtime_defaults.get("pretrained_weights", "pretrain_model/yolov12m.pt")
        self.pretrained_weights = resolve_project_path(
            pretrained,
            "HFSA_CLASSIFICATION_PRETRAINED_WEIGHTS",
        )
        self.device = os.environ.get("HFSA_CLASSIFICATION_DEVICE", "auto").strip() or "auto"
        self.imgsz = int(config.runtime_defaults.get("imgsz", 640))
        self.topk = int(config.runtime_defaults.get("topk", 5))
        self._predictor_factory = predictor_factory or _default_predictor_factory
        self._predictor: Any | None = None
        self._lock = threading.Lock()

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"Scene classification checkpoint not found: {self.checkpoint}")
            if not self.pretrained_weights.is_file():
                raise FileNotFoundError(f"YOLOv12 pretrained weights not found: {self.pretrained_weights}")
            self._predictor = self._predictor_factory(
                checkpoint=self.checkpoint,
                pretrained_weights=self.pretrained_weights,
                device=self.device,
                imgsz=self.imgsz,
            )
        return self._predictor

    def predict(self, image: Any) -> dict[str, Any]:
        decoded_image = decode_data_url(image)
        with self._lock:
            result = self._get_predictor().predict(decoded_image, topk=self.topk)
        predictions = list(result.get("predictions", []))
        if not predictions:
            raise RuntimeError("Scene classification returned no class predictions.")
        for item in predictions:
            if not {"class_id", "class_name", "probability"}.issubset(item):
                raise RuntimeError("Scene classification prediction schema is incomplete.")
        top1 = predictions[0]
        return {
            "summary": {
                "top1_class": str(top1["class_name"]),
                "top1_probability": float(top1["probability"]),
                "latency_ms": float(result.get("latency_ms", 0.0)),
                "original_size": list(result.get("original_size", [])),
                "imgsz": int(result.get("imgsz", self.imgsz)),
            },
            "predictions": predictions,
        }

    def close(self) -> None:
        predictor, self._predictor = self._predictor, None
        release_model_resources(predictor)
