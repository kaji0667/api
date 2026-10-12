"""Official-only detection using the submitted test's text/phrase-map contract."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from api_task_text import TaskInputError, detection_text
from grounding_spatial import parse_absolute_position

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from text_encoder.matching import assign_by_score_matrix, score_matrix_from_phrase_maps
from text_encoder.phrase_classifier import classify_phrase, get_phrase_type_weight
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder, sanitize_text
from ultralytics.utils.ops import non_max_suppression


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    value = str(device).strip().lower()
    if value in {"", "auto"}:
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


class NativeTextEncoder:
    """Share one unchanged OpenCLIP model with task-specific input preparation."""

    COUNTING_TEMPLATE = "a satellite remote sensing photo of {class_name}"
    CLASS_ALIASES = {
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

    def __init__(self, device: str | torch.device = "auto",
                 model_name: str = "ViT-L-14", pretrained: str = "openai") -> None:
        self.device = resolve_device(device)
        self.encoder = OpenCLIPTextEncoder(
            model_name=model_name, pretrained=pretrained, device=str(self.device),
            precision="fp32", normalize=True,
        )
        self.tokenizer = self.encoder.tokenizer
        if (getattr(self.tokenizer, "context_length", None) != 77
                or not callable(getattr(self.tokenizer, "encode", None))):
            raise TaskInputError("unsupported_text_tokenizer", "A verifiable 77-token tokenizer is required")
        self._cache: dict[str, torch.Tensor] = {}

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        prepared = [sanitize_text(text, empty_placeholder="") for text in texts]
        if not prepared or any(not text for text in prepared):
            raise TaskInputError("missing_target", "A nonempty model description is required")
        for text in prepared:
            # SimpleTokenizer.encode excludes the two special tokens. Check the
            # exact final prompt before its callable tokenizer can truncate it.
            if len(self.tokenizer.encode(text)) > 75:
                raise TaskInputError("text_context_exceeded", "Text exceeds the OpenCLIP content-token limit")
        missing = list(dict.fromkeys(text for text in prepared if text not in self._cache))
        if missing:
            features = self.encoder.encode(texts=missing, batch_size=256, return_tokens=False)
            if (not isinstance(features, torch.Tensor) or features.shape != (len(missing), 768)
                    or not torch.isfinite(features).all()):
                raise TaskInputError("invalid_text_embedding", "Invalid OpenCLIP description embeddings")
            features = features.detach().float().cpu()
            norms = features.norm(p=2, dim=-1, keepdim=True)
            if bool((norms <= 0).any()):
                raise TaskInputError("invalid_text_embedding", "Empty OpenCLIP description embeddings")
            # OpenCLIPTextEncoder already applies the original one-pass
            # normalization. Re-normalizing changes the cached numeric values.
            for text, feature in zip(missing, features):
                self._cache[text] = feature.clone()
        return torch.stack([self._cache[text] for text in prepared]).clone()

    def encode_descriptions(self, texts: Sequence[str]) -> torch.Tensor:
        """One pooled, normalized vector per bare object description, as cached offline."""
        return self._encode(texts)

    @classmethod
    def counting_prompt(cls, query: str) -> str:
        cleaned = " ".join(str(query).replace("-", " ").replace("_", " ").strip().lower().split())
        if not cleaned:
            raise TaskInputError("missing_target", "A nonempty counting target is required")
        target = cls.CLASS_ALIASES.get(cleaned, cleaned)
        return cls.COUNTING_TEMPLATE.format(class_name=target)

    def encode_counting(self, query: str) -> torch.Tensor:
        return self._encode([self.counting_prompt(query)])


@dataclass(frozen=True)
class NativeDetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    original_size: tuple[int, int]
    grounding_index: int | None
    model_prompt: str
    diagnostics: dict[str, Any] | None = None
    phrase_scores: np.ndarray | None = None
    extra_phrase_scores: np.ndarray | None = None

    @property
    def detection_count(self) -> int:
        return int(len(self.boxes))


class NativeDetection:
    """Single-description inference without test labels, GT matching or Web Predictors."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "auto",
                 encoder: NativeTextEncoder | None = None, *, max_image_size: int = 800,
                 spatial_weight: float = .15, candidate_confidence: float = .25) -> None:
        if type(max_image_size) is not int or max_image_size < 800 or max_image_size % 32:
            raise ValueError("Detection max_image_size must be a multiple of 32, at least 800")
        self.max_image_size = max_image_size
        if (isinstance(spatial_weight, bool) or not isinstance(spatial_weight, (int, float))
                or not math.isfinite(spatial_weight) or not 0 <= spatial_weight <= 1):
            raise ValueError("spatial_weight must be a finite number between 0 and 1")
        self.spatial_weight = float(spatial_weight)
        self.candidate_confidence = candidate_confidence
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError("Native detection checkpoint is missing")
        self.device = resolve_device(device)
        # The original submitted detection test uses this PyTorch backend and
        # explicitly sends augment=False plus cached object-aligned conditions.
        self.backend = AutoBackend(weights=str(self.checkpoint), device=self.device,
                                   dnn=False, fp16=False)
        if not (self.backend.pt or self.backend.nn_module):
            raise TaskInputError("unsupported_detection_backend", "Detection requires a PyTorch checkpoint")
        self.model = self.backend.model
        self.model.eval()
        self.encoder = encoder if encoder is not None else NativeTextEncoder(device=self.device)

    @property
    def candidate_confidence(self) -> float:
        return self._candidate_confidence

    @candidate_confidence.setter
    def candidate_confidence(self, value: float) -> None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value < 1):
            raise ValueError("candidate_confidence must be a finite number strictly between 0 and 1")
        self._candidate_confidence = float(value)

    @staticmethod
    def _image(image: Image.Image | str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise TaskInputError("invalid_detection_image", "Expected a uint8 BGR image")
            result = np.ascontiguousarray(image)
        elif isinstance(image, Image.Image):
            result = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image, (str, Path)):
            # Match BaseDataset.load_image, including its OpenCV file decoding.
            result = cv2.imread(str(image))
        else:
            raise TaskInputError("invalid_detection_image", "Unsupported detection image input")
        if result is None or min(result.shape[:2]) <= 0:
            raise TaskInputError("invalid_detection_image", "Empty detection image")
        return result

    def image_size_for(self, original_hw: tuple[int, int]) -> int:
        # Small source-test images retain imgsz=800. Larger API assets use the
        # original evaluator's imgsz parameter, bounded by the deployment cap.
        return min(self.max_image_size, max(800, math.ceil(max(original_hw) / 32) * 32))

    @staticmethod
    def _preprocess(image: np.ndarray, stride: int = 32, image_size: int = 800) -> tuple[torch.Tensor, tuple[Any, Any]]:
        height, width = image.shape[:2]
        # The original validation dataset first resizes the long side with
        # ceil rounding, then pads to its batch=1 rect shape (pad=0.5).
        ratio = image_size / max(height, width)
        if ratio != 1:
            new_size = (min(math.ceil(width * ratio), image_size), min(math.ceil(height * ratio), image_size))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        resized_h, resized_w = image.shape[:2]
        rect_shape = np.ceil(
            np.array([height, width], dtype=np.float64) / max(height, width) * image_size / stride + .5
        ).astype(int) * stride
        left = int(round((int(rect_shape[1]) - resized_w) / 2 - .1))
        top = int(round((int(rect_shape[0]) - resized_h) / 2 - .1))
        image = LetterBox(new_shape=tuple(rect_shape), scaleup=False, auto=False,
                          stride=stride)(image=image)
        rgb_chw = image[:, :, ::-1].transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb_chw)).float().unsqueeze(0) / 255
        return tensor, ((resized_h / height, resized_w / width), (left, top))

    @staticmethod
    def _phrase_maps(model: Any) -> list[torch.Tensor]:
        output = getattr(model, "_last_text_outputs", None)
        levels = output.get("phrase_logits") if isinstance(output, dict) else None
        if not isinstance(levels, list) or not levels:
            raise TaskInputError("detection_phrase_maps_missing", "No description-to-image phrase maps")
        if any(not isinstance(level, torch.Tensor) or level.ndim != 4
               or level.shape[0] != 1 or level.shape[1] != 1
               or min(level.shape[2:]) <= 0 or not torch.isfinite(level).all()
               for level in levels):
            raise TaskInputError("invalid_detection_phrase_maps", "Invalid description-to-image phrase maps")
        return levels

    @staticmethod
    def _restore_boxes(boxes: torch.Tensor, original_hw: tuple[int, int],
                       ratio_pad: tuple[Any, Any]) -> torch.Tensor:
        """Remove rect padding and invert the actual ceil resize on each axis."""
        try:
            height, width = original_hw
            gain_hw, padding_xy = ratio_pad
            gain_h, gain_w = (float(value) for value in gain_hw)
            left, top = (float(value) for value in padding_xy)
        except (TypeError, ValueError, OverflowError):
            raise TaskInputError("invalid_detection_geometry", "Invalid coordinate recovery metadata") from None
        if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or value <= 0 for value in (height, width))
                or not all(math.isfinite(value) for value in (gain_h, gain_w, left, top))
                or min(gain_h, gain_w) <= 0 or min(left, top) < 0
                or not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4
                or not boxes.is_floating_point() or not torch.isfinite(boxes).all()):
            raise TaskInputError("invalid_detection_geometry", "Invalid coordinate recovery inputs")
        restored = boxes.clone()
        # BaseDataset ceil-rounds each resized dimension separately. Using only
        # gain_h for x, as the original evaluator does, is not an exact inverse.
        restored[:, [0, 2]] = ((restored[:, [0, 2]] - left) / gain_w).clamp(0, width)
        restored[:, [1, 3]] = ((restored[:, [1, 3]] - top) / gain_h).clamp(0, height)
        return restored

    def _position_adjusted_scores(self, score: torch.Tensor, boxes: torch.Tensor,
                                  original_hw: tuple[int, int], description: str) -> torch.Tensor:
        """Add a bounded image-frame prior; never remove or invent candidates."""
        if self.spatial_weight == 0:
            return score
        position = parse_absolute_position(description)
        if position is None:
            return score
        height, width = original_hw
        centers = ((boxes[:, :2] + boxes[:, 2:]) * .5
                   / boxes.new_tensor([width, height])).clamp(0, 1)
        axis_scores = []
        for axis, direction in enumerate((position.horizontal, position.vertical)):
            if direction is None:
                continue
            coordinate = centers[:, axis]
            if direction == -1:
                axis_scores.append(1 - coordinate)
            elif direction == 1:
                axis_scores.append(coordinate)
            else:
                axis_scores.append(1 - 2 * (coordinate - .5).abs())
        prior = torch.stack(axis_scores).mean(dim=0)
        nonempty = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        prior = prior * nonempty.to(dtype=prior.dtype)
        return score + self.spatial_weight * prior.unsqueeze(0)

    @staticmethod
    def _score_boxes_for_input(score_boxes_xyxy: Any, original_hw: tuple[int, int],
                               ratio_pad: tuple[Any, Any]) -> np.ndarray:
        """Validate external original-image boxes and apply the real per-axis resize."""
        try:
            source = (score_boxes_xyxy.detach().cpu().numpy()
                      if isinstance(score_boxes_xyxy, torch.Tensor) else np.asarray(score_boxes_xyxy))
            if source.ndim != 2 or source.shape[1] != 4 or source.dtype.kind not in "iuf":
                raise ValueError("Expected real numeric boxes shaped [N, 4]")
            boxes = source.astype(np.float64, copy=True)
            height, width = original_hw
            gain_hw, padding_xy = ratio_pad
            gain_h, gain_w = (float(value) for value in gain_hw)
            left, top = (float(value) for value in padding_xy)
        except (TypeError, ValueError, OverflowError, RuntimeError):
            raise TaskInputError("invalid_detection_geometry", "Invalid external scoring boxes or geometry") from None
        if (not np.isfinite(boxes).all()
                or not all(math.isfinite(value) for value in (gain_h, gain_w, left, top))
                or min(gain_h, gain_w) <= 0 or min(left, top) < 0
                or (boxes < 0).any() or (boxes > [width, height, width, height]).any()
                or (boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any()):
            raise TaskInputError("invalid_detection_geometry", "External scoring boxes must have area inside the image")
        with np.errstate(over="ignore", invalid="ignore"):
            mapped = boxes * [gain_w, gain_h, gain_w, gain_h] + [left, top, left, top]
        if not np.isfinite(mapped).all():
            raise TaskInputError("invalid_detection_geometry", "External scoring coordinates must remain finite")
        return mapped

    def predict(self, image: Image.Image | str | Path | np.ndarray,
                original_question: str, *, require_grounding: bool = True,
                return_phrase_scores: bool = False,
                score_boxes_xyxy: Any = None) -> NativeDetectionResult:
        """Optionally return raw phrase scores without changing the detection candidates.

        External boxes are original-image xyxy coordinates. They and the local
        NMS candidates are scored in this forward's phrase maps, before the
        optional absolute-position bonus; external boxes never enter assignment.
        """
        if type(return_phrase_scores) is not bool:
            raise TaskInputError("invalid_detection_matching", "return_phrase_scores must be boolean")
        description = sanitize_text(detection_text(original_question), empty_placeholder="")
        candidate_confidence = self.candidate_confidence
        # One description is one object-aligned row, not sentence fragments or
        # a fabricated collection of other objects from the unseen test labels.
        vectors = self.encoder.encode_descriptions([description])
        if not isinstance(vectors, torch.Tensor) or vectors.shape != (1, 768) or not torch.isfinite(vectors).all():
            raise TaskInputError("invalid_text_embedding", "Invalid detection description embedding")
        original = self._image(image)
        tensor, ratio_pad = self._preprocess(
            original, stride=int(self.backend.stride), image_size=self.image_size_for(original.shape[:2]),
        )
        extra_input_boxes = (self._score_boxes_for_input(score_boxes_xyxy, original.shape[:2], ratio_pad)
                             if score_boxes_xyxy is not None else None)
        tensor = tensor.to(self.device)
        vectors = vectors.to(device=self.device, dtype=tensor.dtype).unsqueeze(0)
        mask = torch.ones((1, 1), device=self.device, dtype=torch.bool)
        weight = get_phrase_type_weight(classify_phrase(description))
        phrase_weight = torch.tensor([[weight]], device=self.device, dtype=tensor.dtype)
        dependency = torch.ones((1, 1, 1), device=self.device, dtype=tensor.dtype)
        with torch.inference_mode():
            raw = self.model(tensor, augment=False, txt_vec=vectors,
                             txt_token_mask=mask, txt_phrase_weight=phrase_weight,
                             txt_dependency_adj=dependency)
            diagnostic = None
            if os.environ.get("HFSA_API_DIAGNOSTICS") == "1":
                prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
                # Detection checkpoints expose [batch, xywh+class, anchors].
                # Observe the same confidence threshold that NMS will apply.
                if (isinstance(prediction, torch.Tensor) and prediction.ndim == 3
                        and prediction.shape[0] == 1 and prediction.shape[1] >= 5
                        and prediction.shape[2] > 0):
                    confidence = prediction[0, 4:].amax(dim=0)
                    if torch.isfinite(confidence).all():
                        diagnostic = {
                            "original_hw": list(original.shape[:2]),
                            "input_hw": list(tensor.shape[2:]),
                            "confidence_threshold": candidate_confidence,
                            "raw_anchor_count": int(confidence.numel()),
                            "max_confidence": float(confidence.max()),
                            "pre_nms_above_threshold": int((confidence > candidate_confidence).sum()),
                        }
            detections = non_max_suppression(raw, conf_thres=candidate_confidence, iou_thres=.7,
                                            multi_label=False, agnostic=True, max_det=300)[0]
            if (not isinstance(detections, torch.Tensor) or detections.ndim != 2
                    or detections.shape[1] < 5 or not torch.isfinite(detections).all()):
                raise TaskInputError("invalid_detection_result", "Invalid native detection candidates")
            if diagnostic is not None:
                diagnostic["nms_candidate_count"] = len(detections)
            boxes = detections[:, :4].clone()
            if len(boxes):
                boxes = self._restore_boxes(boxes, original.shape[:2], ratio_pad)
            selected = None
            phrase_scores = np.empty((0,), dtype=np.float32) if return_phrase_scores else None
            extra_phrase_scores = np.empty((0,), dtype=np.float32) if extra_input_boxes is not None else None
            extra_count = len(extra_input_boxes) if extra_input_boxes is not None else 0
            need_scores = require_grounding or return_phrase_scores or extra_count > 0
            if need_scores and (len(detections) or extra_count):
                scoring_boxes = detections[:, :4]
                if extra_count:
                    external = torch.as_tensor(extra_input_boxes, device=detections.device, dtype=detections.dtype)
                    scoring_boxes = torch.cat((scoring_boxes, external), dim=0)
                score = score_matrix_from_phrase_maps(
                    phrase_logits_levels=self._phrase_maps(self.model), image_index=0,
                    token_indices=torch.zeros((1,), device=detections.device, dtype=torch.long),
                    candidate_boxes_xyxy=scoring_boxes, input_hw=tuple(tensor.shape[2:]),
                )
                if score.shape != (1, len(detections) + extra_count) or not torch.isfinite(score).all():
                    raise TaskInputError("invalid_detection_matching", "Invalid description matching scores")
                if return_phrase_scores:
                    phrase_scores = score[0, :len(detections)].detach().cpu().numpy().astype(np.float32, copy=True)
                if extra_count:
                    extra_phrase_scores = score[0, len(detections):].detach().cpu().numpy().astype(np.float32, copy=True)
                score = score[:, :len(detections)]
            if require_grounding and len(detections):
                score = self._position_adjusted_scores(score, boxes, original.shape[:2], description)
                rows, columns = assign_by_score_matrix(score, temperature=.7)
                if rows.numel() != 1 or columns.numel() != 1 or int(rows[0]) != 0:
                    raise TaskInputError("invalid_detection_matching", "No unique description matching assignment")
                selected = int(columns[0])
                if not 0 <= selected < len(detections):
                    raise TaskInputError("invalid_detection_matching", "Invalid description matching index")
        return NativeDetectionResult(
            boxes=boxes.detach().cpu().numpy().astype(np.float32, copy=False),
            scores=detections[:, 4].detach().cpu().numpy().astype(np.float32, copy=False),
            original_size=tuple(int(x) for x in original.shape[:2]),
            grounding_index=selected, model_prompt=description,
            diagnostics=diagnostic, phrase_scores=phrase_scores,
            extra_phrase_scores=extra_phrase_scores,
        )
