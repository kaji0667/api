"""Single-image, text-guided referring-segmentation inference.

This module is the task-owned inference boundary used by the future manual
task router. It accepts exactly the inputs required by referring segmentation:
one image and one referring expression. Other tasks keep their own input and
output contracts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ultralytics.nn import SemanticSegmentationModel
from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ImageInput = str | Path | Image.Image | np.ndarray


def _resolve_existing_path(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(PROJECT_ROOT / candidate)
    for item in candidates:
        if item.is_file():
            return item.resolve()
    raise FileNotFoundError(f"{label} not found: {path}")


def _resolve_model_yaml(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.is_file():
        return str(project_candidate.resolve())
    return str(path)


def _load_rgb_image(image: ImageInput) -> tuple[np.ndarray, str]:
    source = "<memory>"
    if isinstance(image, (str, Path)):
        image_path = _resolve_existing_path(image, "Inference image")
        source = str(image_path)
        with Image.open(image_path) as handle:
            array = np.asarray(handle.convert("RGB"))
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("NumPy image input must have shape [H, W, 3] or [H, W, 4].")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size and float(array.max()) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image input type: {type(image).__name__}")

    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"Expected a non-empty RGB image, got shape {tuple(array.shape)}")
    return np.ascontiguousarray(array, dtype=np.uint8), source


def _safe_stem(value: str) -> str:
    cleaned = "_".join(str(value).replace("/", " ").replace("\\", " ").split())
    return cleaned or "refseg_prediction"


@dataclass
class RefSegPrediction:
    """In-memory prediction plus serializable metadata and image artifacts."""

    mask: np.ndarray
    probability: np.ndarray
    overlay: np.ndarray
    prompt: str
    source_image: str
    checkpoint: str
    threshold: float
    input_size: int
    original_height: int
    original_width: int
    latency_ms: float

    def summary(self) -> dict[str, Any]:
        foreground_pixels = int(self.mask.sum())
        total_pixels = int(self.mask.size)
        return {
            "task": "referring_segmentation",
            "source_image": self.source_image,
            "prompt": self.prompt,
            "checkpoint": self.checkpoint,
            "threshold": float(self.threshold),
            "input_size": int(self.input_size),
            "original_size": [int(self.original_height), int(self.original_width)],
            "foreground_pixels": foreground_pixels,
            "foreground_ratio": foreground_pixels / max(total_pixels, 1),
            "latency_ms": float(self.latency_ms),
        }

    def save(self, save_dir: str | Path, stem: str = "") -> dict[str, str]:
        output_dir = Path(save_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if stem:
            output_stem = _safe_stem(stem)
        elif self.source_image != "<memory>":
            output_stem = _safe_stem(Path(self.source_image).stem)
        else:
            output_stem = "refseg_prediction"

        mask_path = output_dir / f"{output_stem}_mask.png"
        probability_path = output_dir / f"{output_stem}_probability.png"
        overlay_path = output_dir / f"{output_stem}_overlay.png"
        result_path = output_dir / f"{output_stem}_result.json"

        Image.fromarray(self.mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        Image.fromarray(np.clip(self.probability * 255.0, 0, 255).astype(np.uint8), mode="L").save(
            probability_path
        )
        Image.fromarray(self.overlay, mode="RGB").save(overlay_path)

        report = self.summary()
        report["artifacts"] = {
            "mask": str(mask_path),
            "probability": str(probability_path),
            "overlay": str(overlay_path),
        }
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        return {
            "mask": str(mask_path),
            "probability": str(probability_path),
            "overlay": str(overlay_path),
            "result": str(result_path),
        }


class RefSegPredictor:
    """Load one RefSeg checkpoint and predict masks for image-text requests."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        model_yaml: str = "",
        imgsz: int | None = None,
        threshold: float | None = None,
        text_model_name: str = "",
        text_pretrained: str = "",
        text_precision: str = "",
    ) -> None:
        self.checkpoint_path = _resolve_existing_path(checkpoint, "RefSeg checkpoint")
        self.device = self._resolve_device(device)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise ValueError("RefSeg checkpoint must contain a model state dictionary.")

        checkpoint_args = dict(payload.get("args") or {})
        checkpoint_data = dict(payload.get("data") or {})
        checkpoint_metrics = dict(payload.get("metrics") or {})
        self.checkpoint_epoch = int(payload.get("epoch", 0))
        self.imgsz = int(imgsz if imgsz is not None else checkpoint_args.get("imgsz", 512))
        if self.imgsz <= 0:
            raise ValueError("imgsz must be positive.")

        stored_threshold = checkpoint_metrics.get("best_threshold", 0.5)
        self.threshold = float(stored_threshold if threshold is None else threshold)
        if not math.isfinite(self.threshold) or not 0.0 < self.threshold < 1.0:
            raise ValueError(f"Mask threshold must be between 0 and 1, got {self.threshold}")

        resolved_model_yaml = model_yaml or str(
            checkpoint_args.get("model") or "ultralytics/cfg/models/v12/yolov12m-semseg.yaml"
        )
        self.model_yaml = _resolve_model_yaml(resolved_model_yaml)
        num_classes = int(checkpoint_data.get("nc", 1))
        if num_classes <= 0:
            raise ValueError(f"Checkpoint data.nc must be positive, got {num_classes}")

        self.model = SemanticSegmentationModel(self.model_yaml, ch=3, nc=num_classes, verbose=False)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model = self.model.to(self.device).eval()

        self.text_model_name = text_model_name or str(checkpoint_args.get("text_model_name") or "ViT-L-14")
        self.text_pretrained = text_pretrained or str(checkpoint_args.get("text_pretrained") or "openai")
        self.text_precision = text_precision or str(checkpoint_args.get("text_precision") or "fp32")
        self.text_encoder = OpenCLIPTextEncoder(
            model_name=self.text_model_name,
            pretrained=self.text_pretrained,
            device=str(self.device),
            precision=self.text_precision,
            normalize=True,
        )
        if int(self.text_encoder.embedding_dim) != 768:
            raise ValueError(
                f"OpenCLIP text dim is {int(self.text_encoder.embedding_dim)}, but TextPromptSegment expects 768."
            )

        del payload
        gc.collect()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        value = str(device or "auto").strip().lower()
        if value == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        return resolved

    def _prepare_image(self, image: ImageInput) -> tuple[np.ndarray, torch.Tensor, str]:
        original, source = _load_rgb_image(image)
        resized = Image.fromarray(original, mode="RGB").resize(
            (self.imgsz, self.imgsz), Image.Resampling.BILINEAR
        )
        resized_array = np.asarray(resized).copy()
        tensor = torch.from_numpy(resized_array).permute(2, 0, 1).contiguous().float() / 255.0
        return original, tensor.unsqueeze(0).to(self.device), source

    @staticmethod
    def _overlay(original: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        overlay = original.copy()
        color = np.array([255, 64, 64], dtype=np.float32)
        selected = mask.astype(bool)
        if selected.any():
            overlay[selected] = np.clip(
                overlay[selected].astype(np.float32) * (1.0 - alpha) + color * alpha,
                0,
                255,
            ).astype(np.uint8)
        return overlay

    def predict(self, image: ImageInput, text: str) -> RefSegPrediction:
        prompt = str(text).strip()
        if not prompt:
            raise ValueError("Referring segmentation requires a non-empty text description.")

        original, image_tensor, source = self._prepare_image(image)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        text_features, text_token_mask = self.text_encoder.encode(
            [prompt], batch_size=1, return_tokens=True
        )
        text_features = text_features.to(device=self.device, dtype=torch.float32)
        text_token_mask = text_token_mask.to(device=self.device, dtype=torch.bool)
        if text_features.ndim != 3 or text_features.shape[0] != 1 or text_features.shape[-1] != 768:
            raise RuntimeError(
                f"Online OpenCLIP token features must be [1, T, 768], got {tuple(text_features.shape)}"
            )

        with torch.inference_mode():
            predictions = self.model(
                image_tensor,
                text_embedding=text_features,
                text_token_mask=text_token_mask,
            )
            logits = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
            if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
                raise RuntimeError(f"RefSeg model must return [1, 1, H, W] logits, got {tuple(logits.shape)}")
            probability = logits.sigmoid()
            probability = F.interpolate(
                probability,
                size=(int(original.shape[0]), int(original.shape[1])),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        probability_array = probability.detach().cpu().float().numpy()
        mask = probability_array > self.threshold
        return RefSegPrediction(
            mask=mask,
            probability=probability_array,
            overlay=self._overlay(original, mask),
            prompt=prompt,
            source_image=source,
            checkpoint=str(self.checkpoint_path),
            threshold=self.threshold,
            input_size=self.imgsz,
            original_height=int(original.shape[0]),
            original_width=int(original.shape[1]),
            latency_ms=latency_ms,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-image text-guided referring segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Full RefSeg checkpoint containing model state.")
    parser.add_argument("--image", required=True, help="Input remote-sensing image.")
    parser.add_argument("--text", required=True, help="Referring expression describing the target.")
    parser.add_argument("--save-dir", default="runs/semseg/inference", help="Output directory.")
    parser.add_argument("--output-stem", default="", help="Optional output filename prefix.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--model", default="", help="Optional model YAML override; checkpoint value is preferred.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional square input-size override.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional mask threshold override.")
    parser.add_argument("--text-model-name", default="", help="Optional OpenCLIP model override.")
    parser.add_argument("--text-pretrained", default="", help="Optional OpenCLIP pretrained-tag override.")
    parser.add_argument("--text-precision", default="", help="Optional OpenCLIP precision override.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predictor = RefSegPredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        model_yaml=args.model,
        imgsz=args.imgsz,
        threshold=args.threshold,
        text_model_name=args.text_model_name,
        text_pretrained=args.text_pretrained,
        text_precision=args.text_precision,
    )
    prediction = predictor.predict(args.image, args.text)
    artifacts = prediction.save(args.save_dir, stem=args.output_stem)
    report = prediction.summary()
    report["artifacts"] = artifacts
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
