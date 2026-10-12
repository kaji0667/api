"""Object-counting inference built on the shared text-guided detection runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tasks.detection.inference import (
    TextGuidedDetectionPredictor,
    TextGuidedDetector,
    TextGuidedImagePreprocessor,
    TextPromptEncoder,
)


CountingImagePreprocessor = TextGuidedImagePreprocessor
CountingPromptEncoder = TextPromptEncoder


class ObjectCounter:
    """Preserve the teammate's tensor-level counting interface."""

    def __init__(self, model_instance: Any, prompt_encoder: CountingPromptEncoder, device: str | torch.device) -> None:
        self.runtime = TextGuidedDetector(model_instance, prompt_encoder, device)
        self.device = self.runtime.device
        self.model = self.runtime.model
        self.prompt_encoder = self.runtime.prompt_encoder

    def count(
        self,
        image_tensor: torch.Tensor,
        target_class: str,
        conf_thres: float = 0.15,
        iou_thres: float = 0.50,
        max_det: int = 300,
    ) -> dict[str, Any]:
        result = self.runtime.detect(image_tensor, target_class, conf_thres, iou_thres, max_det)
        return {
            "target_class": str(target_class),
            "pred_count": int(len(result["boxes"])),
            "pred_boxes": result["boxes"],
            "pred_scores": result["scores"],
        }


@dataclass(frozen=True)
class CountingPrediction:
    target_class: str
    count: int
    boxes: np.ndarray
    scores: np.ndarray
    visualization: np.ndarray
    latency_ms: float
    original_size: tuple[int, int]


class CountingPredictor:
    """Checkpoint-owning single-image counting boundary for the Web adapter."""

    def __init__(self, checkpoint: str | Path, **kwargs: Any) -> None:
        self.detector = TextGuidedDetectionPredictor(checkpoint=checkpoint, **kwargs)

    def predict(
        self,
        image: Any,
        target_class: str,
        conf_thres: float = 0.15,
        iou_thres: float = 0.50,
        max_det: int = 300,
    ) -> CountingPrediction:
        result = self.detector.predict(
            image,
            target_class,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )
        return CountingPrediction(
            target_class=result.text,
            count=result.detection_count,
            boxes=result.boxes,
            scores=result.scores,
            visualization=result.visualization,
            latency_ms=result.latency_ms,
            original_size=result.original_size,
        )
