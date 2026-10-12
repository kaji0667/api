"""Native official-API RefSeg execution using the submitted evaluation runtime.

This module owns API input/output adaptation. It does not import the manual Web
adapter or its Predictor, and does not change the model or OpenCLIP implementation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from api_task_text import TaskInputError, grounding_text
from tasks.refseg.runtime import (
    PROJECT_ROOT, predict_with_optional_text, resolve_checkpoint_path, resolve_existing_path,
)


def _model_factory(model_yaml: str, **kwargs: Any) -> Any:
    from ultralytics.nn import SemanticSegmentationModel

    return SemanticSegmentationModel(model_yaml, **kwargs)


def _text_encoder_factory(**kwargs: Any) -> Any:
    from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder

    return OpenCLIPTextEncoder(**kwargs)


@dataclass(frozen=True)
class NativeRefSegPrediction:
    mask: np.ndarray
    probability: np.ndarray
    threshold: float
    original_size: tuple[int, int]
    model_prompt: str
    input_size: int
    latency_ms: float


class NativeRefSegPredictor:
    """Serve the original strict RefSeg checkpoint with native-size binary masks."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "auto",
        model_yaml: str = "",
        *,
        imgsz: int | None = None,
        model_factory: Callable[..., Any] | None = None,
        text_encoder_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.checkpoint = resolve_checkpoint_path(checkpoint)
        if self.checkpoint.name != "best_raw.pt":
            raise TaskInputError("invalid_refseg_checkpoint", "RefSeg requires its original best_raw.pt checkpoint")
        requested_device = str(device).strip().lower()
        if requested_device in {"", "auto"}:
            requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise TaskInputError("invalid_refseg_checkpoint", "RefSeg checkpoint must contain the complete model state")
        checkpoint_args = dict(payload.get("args") or {})
        checkpoint_data = dict(payload.get("data") or {})
        checkpoint_metrics = dict(payload.get("metrics") or {})
        # Explicit CLI/API overrides take precedence. The formal checkpoint and
        # original test script use 512; its saved value is retained when present.
        self.imgsz = int(imgsz if imgsz is not None else checkpoint_args.get("imgsz", 512))
        if self.imgsz <= 0:
            raise TaskInputError("invalid_refseg_configuration", "RefSeg image size must be positive")
        self.threshold = float(checkpoint_metrics.get("best_threshold", 0.5))
        if not math.isfinite(self.threshold) or not 0.0 < self.threshold < 1.0:
            raise TaskInputError("invalid_refseg_checkpoint", "RefSeg validation threshold is invalid")
        resolved_yaml = str(model_yaml).strip() or str(
            checkpoint_args.get("model") or "ultralytics/cfg/models/v12/yolov12m-semseg.yaml"
        )
        yaml_path = resolve_existing_path(resolved_yaml)
        if yaml_path is None and not Path(resolved_yaml).is_absolute():
            yaml_path = resolve_existing_path(PROJECT_ROOT / resolved_yaml)
        self.model_yaml = str(yaml_path.resolve()) if yaml_path is not None else resolved_yaml
        num_classes = int(checkpoint_data.get("nc", 1))
        if num_classes <= 0:
            raise TaskInputError("invalid_refseg_checkpoint", "RefSeg checkpoint class count is invalid")

        make_model = model_factory or _model_factory
        self.model = make_model(self.model_yaml, ch=3, nc=num_classes, verbose=False)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval()
        make_encoder = text_encoder_factory or _text_encoder_factory
        self.text_encoder = make_encoder(
            model_name=str(checkpoint_args.get("text_model_name") or "ViT-L-14"),
            pretrained=str(checkpoint_args.get("text_pretrained") or "openai"),
            device=str(self.device),
            precision="fp32",
            normalize=True,
        )
        if int(self.text_encoder.embedding_dim) != 768:
            raise TaskInputError("invalid_refseg_text_encoder", "RefSeg requires the original 768-dimensional text features")

    @staticmethod
    def _load_rgb(image: Image.Image | str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        elif isinstance(image, (str, Path)):
            image_path = resolve_existing_path(image)
            if image_path is None:
                raise TaskInputError("invalid_refseg_image", "RefSeg image is unavailable")
            # Match RRSISDRefSegDataset._read_image, including the JPEG decoder.
            decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if decoded is None:
                raise TaskInputError("invalid_refseg_image", "RefSeg image is unavailable")
            array = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] not in {3, 4}:
                raise TaskInputError("invalid_refseg_image", "RefSeg requires a three-channel RGB image")
            array = array[:, :, :3]
            if array.dtype != np.uint8:
                if not np.isfinite(array).all():
                    raise TaskInputError("invalid_refseg_image", "RefSeg image values must be finite")
                if array.size and np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            raise TaskInputError("invalid_refseg_image", "RefSeg image input is unsupported")
        if array.ndim != 3 or array.shape[2] != 3 or not array.shape[0] or not array.shape[1]:
            raise TaskInputError("invalid_refseg_image", "RefSeg requires a non-empty RGB image")
        return np.ascontiguousarray(array)

    def _check_text_budget(self, prompt: str) -> None:
        # Match OpenCLIPTextEncoder.encode's input normalization before counting.
        from ultralytics.nn.modules.text_backbone import sanitize_text

        tokenizer = self.text_encoder.tokenizer
        context_length = int(getattr(tokenizer, "context_length", 77))
        full_token_count = len(tokenizer.encode(sanitize_text(prompt, empty_placeholder=""))) + 2
        if full_token_count > context_length:
            raise TaskInputError("text_context_exceeded", "RefSeg target description exceeds the text encoder context")

    def predict(
        self, image: Image.Image | str | Path | np.ndarray, original_question: str
    ) -> NativeRefSegPrediction:
        started = time.perf_counter()
        prompt = grounding_text(original_question)
        self._check_text_budget(prompt)
        original = self._load_rgb(image)
        # Original RefSeg validation uses OpenCV INTER_LINEAR for RGB pixels.
        resized = cv2.resize(original, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float() / 255.0
        tensor = tensor.unsqueeze(0).to(device=self.device, dtype=torch.float32)
        features, token_mask = self.text_encoder.encode([prompt], batch_size=1, return_tokens=True)
        features = features.to(device=self.device, dtype=torch.float32)
        token_mask = token_mask.to(device=self.device, dtype=torch.bool)
        if (
            features.ndim != 3 or features.shape[0] != 1 or features.shape[-1] != 768
            or tuple(token_mask.shape) != tuple(features.shape[:2])
            or not bool(token_mask.any()) or not bool(torch.isfinite(features).all())
        ):
            raise TaskInputError("invalid_refseg_text_features", "RefSeg token features or mask are invalid")
        with torch.inference_mode():
            output = predict_with_optional_text(
                self.model,
                {"img": tensor, "text_embedding": features, "text_token_mask": token_mask},
                text_queries=True,
            )
            logits = output[0] if isinstance(output, (list, tuple)) else output
            if (
                not isinstance(logits, torch.Tensor) or logits.ndim != 4
                or tuple(logits.shape[:2]) != (1, 1) or not bool(torch.isfinite(logits).all())
            ):
                raise TaskInputError("invalid_refseg_result", "RefSeg must return finite single-target logits")
            # Original evaluation thresholds its model grid against resized GT.
            # The API needs native coordinates, so restore *probabilities* first
            # and apply that same validation-selected threshold only afterwards.
            probability = F.interpolate(
                logits.float().sigmoid(), size=original.shape[:2], mode="bilinear", align_corners=False
            )[0, 0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        probability_array = probability.detach().cpu().numpy().astype(np.float32, copy=False)
        return NativeRefSegPrediction(
            mask=probability_array > self.threshold,
            probability=probability_array,
            threshold=self.threshold,
            original_size=(int(original.shape[0]), int(original.shape[1])),
            model_prompt=prompt,
            input_size=self.imgsz,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
