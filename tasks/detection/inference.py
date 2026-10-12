"""Reusable single-image inference for the existing text-guided detector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.utils.ops import non_max_suppression, scale_boxes


ModelFactory = Callable[[Path], Any]
PromptEncoderFactory = Callable[..., Any]
DetectorFactory = Callable[[Any, Any, torch.device], Any]


def resolve_device(device: str | torch.device) -> torch.device:
    requested = str(device).strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


class TextGuidedImagePreprocessor:
    """Apply the existing 800-square letterbox and normalization path."""

    def __init__(self, image_size: int = 800) -> None:
        if int(image_size) <= 0:
            raise ValueError("image_size must be positive")
        self.image_size = int(image_size)

    def __call__(self, image: np.ndarray) -> tuple[torch.Tensor, float, tuple[float, float]]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image shaped [H, W, 3]")
        shape = image.shape[:2]
        ratio = min(self.image_size / shape[0], self.image_size / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = (self.image_size - new_unpad[0]) / 2
        dh = (self.image_size - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = image[:, :, ::-1].transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(tensor)).float() / 255.0
        return tensor.unsqueeze(0), ratio, (dw, dh)


class TextPromptEncoder:
    """Encode and cache one natural-language phrase with the training OpenCLIP domain."""

    CLASS_ALIASES: Mapping[str, str] = {
        "airplane": "airplane",
        "airport": "airport runway",
        "baseball diamond": "baseball field",
        "basketball court": "basketball court",
        "ground track field": "running track sports field",
        "storage tank": "oil storage tank",
        "tennis court": "tennis court",
        "expressway service area": "highway service area",
        "expressway toll station": "toll gate station",
        "vehicle": "small vehicle car",
        "ship": "ship vessel boat",
        "windmill": "wind turbine windmill",
    }

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str | torch.device = "cpu",
        prompt_template: str = "a satellite remote sensing photo of {class_name}",
    ) -> None:
        import open_clip

        self.device = torch.device(device)
        self.prompt_template = str(prompt_template)
        if "{class_name}" not in self.prompt_template:
            raise ValueError("prompt_template must contain '{class_name}'")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.cache: dict[str, torch.Tensor] = {}

    @classmethod
    def clean_prompt(cls, raw_text: str) -> str:
        prompt = " ".join(str(raw_text).replace("-", " ").replace("_", " ").strip().lower().split())
        return cls.CLASS_ALIASES.get(prompt, prompt)

    clean_class_name = clean_prompt

    def encode(self, raw_text: str) -> torch.Tensor:
        cache_key = str(raw_text).strip().lower()
        if not cache_key:
            raise ValueError("Text-guided detection requires a non-empty target description.")
        if cache_key in self.cache:
            return self.cache[cache_key]
        natural_text = self.clean_prompt(cache_key)
        prompt = self.prompt_template.format(class_name=natural_text)
        tokens = self.tokenizer([prompt]).to(self.device)
        with torch.inference_mode():
            feature = self.clip_model.encode_text(tokens)
            feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_vector = feature.unsqueeze(1)
        self.cache[cache_key] = text_vector
        return text_vector


class TextGuidedDetector:
    """Run the senior detector forward contract and class-agnostic NMS."""

    def __init__(self, model_instance: Any, prompt_encoder: TextPromptEncoder, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.model = model_instance.to(self.device).eval()
        self.prompt_encoder = prompt_encoder

    def detect(
        self,
        image_tensor: torch.Tensor,
        text: str,
        conf_thres: float = 0.15,
        iou_thres: float = 0.50,
        max_det: int = 300,
    ) -> dict[str, Any]:
        if not 0.0 <= float(conf_thres) <= 1.0:
            raise ValueError("conf_thres must be between 0 and 1.")
        if not 0.0 <= float(iou_thres) <= 1.0:
            raise ValueError("iou_thres must be between 0 and 1.")
        if int(max_det) <= 0:
            raise ValueError("max_det must be positive.")
        text_vector = self.prompt_encoder.encode(text)
        token_mask = torch.ones((1, text_vector.shape[1]), device=self.device, dtype=torch.bool)
        image_tensor = image_tensor.to(self.device)
        with torch.inference_mode():
            predictions = self.model(image_tensor, txt_vec=text_vector, txt_token_mask=token_mask)
            raw_prediction = predictions[0] if isinstance(predictions, tuple) else predictions
            detections = non_max_suppression(
                raw_prediction,
                conf_thres=float(conf_thres),
                iou_thres=float(iou_thres),
                agnostic=True,
                max_det=int(max_det),
            )[0]
        if detections is not None and len(detections) > 0:
            boxes = detections[:, :4].detach().cpu()
            scores = detections[:, 4].detach().cpu()
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            scores = torch.empty((0,), dtype=torch.float32)
        return {"text": str(text), "boxes": boxes, "scores": scores}


@dataclass(frozen=True)
class TextGuidedDetectionPrediction:
    text: str
    boxes: np.ndarray
    scores: np.ndarray
    visualization: np.ndarray
    latency_ms: float
    original_size: tuple[int, int]

    @property
    def detection_count(self) -> int:
        return int(len(self.boxes))


def _default_model_factory(checkpoint: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(checkpoint)).model


class TextGuidedDetectionPredictor:
    """Own the detector checkpoint, OpenCLIP prompt encoder, preprocessing and visualization."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "auto",
        imgsz: int = 800,
        text_model_name: str = "ViT-L-14",
        text_pretrained: str = "openai",
        prompt_template: str = "a satellite remote sensing photo of {class_name}",
        model_factory: ModelFactory = _default_model_factory,
        prompt_encoder_factory: PromptEncoderFactory = TextPromptEncoder,
        detector_factory: DetectorFactory = TextGuidedDetector,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Text-guided detection checkpoint not found: {self.checkpoint}")
        self.device = resolve_device(device)
        loaded = model_factory(self.checkpoint)
        model = loaded.model if hasattr(loaded, "model") and not isinstance(loaded, torch.nn.Module) else loaded
        prompt_encoder = prompt_encoder_factory(
            model_name=text_model_name,
            pretrained=text_pretrained,
            device=self.device,
            prompt_template=prompt_template,
        )
        self.detector = detector_factory(model, prompt_encoder, self.device)
        self.preprocessor = TextGuidedImagePreprocessor(imgsz)
        self.imgsz = int(imgsz)

    @staticmethod
    def _to_bgr(image: Image.Image | str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected an image array shaped [H, W, 3].")
            return np.ascontiguousarray(image)
        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            with Image.open(image) as source:
                source.load()
                rgb = np.asarray(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _render(image_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray, text: str) -> np.ndarray:
        canvas = image_bgr.copy()
        line_width = max(2, round(min(canvas.shape[:2]) / 300))
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (45, 214, 121), line_width)
            cv2.putText(
                canvas,
                f"{float(score):.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (45, 214, 121),
                max(1, line_width - 1),
                cv2.LINE_AA,
            )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 28, 43), -1)
        cv2.putText(
            canvas,
            f"Target: {text} | Detections: {len(boxes)}",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def predict(
        self,
        image: Image.Image | str | Path | np.ndarray,
        text: str,
        conf_thres: float = 0.15,
        iou_thres: float = 0.50,
        max_det: int = 300,
    ) -> TextGuidedDetectionPrediction:
        prompt = str(text).strip()
        if not prompt:
            raise ValueError("Text-guided detection requires a target description.")
        image_bgr = self._to_bgr(image)
        image_tensor, _, _ = self.preprocessor(image_bgr)
        start = time.perf_counter()
        result = self.detector.detect(image_tensor, prompt, conf_thres, iou_thres, max_det)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        boxes = result["boxes"].clone()
        if len(boxes) > 0:
            scale_boxes(image_tensor.shape[2:], boxes, image_bgr.shape[:2])
        boxes_np = boxes.numpy().astype(np.float32, copy=False)
        scores_np = result["scores"].numpy().astype(np.float32, copy=False)
        visualization = self._render(image_bgr, boxes_np, scores_np, prompt)
        return TextGuidedDetectionPrediction(
            text=prompt,
            boxes=boxes_np,
            scores=scores_np,
            visualization=visualization,
            latency_ms=float(latency_ms),
            original_size=(int(image_bgr.shape[0]), int(image_bgr.shape[1])),
        )
