"""API-owned scene inference using the original classification test primitives."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _create_network(model_yaml: str, classes: int, weights: str, device: str) -> Any:
    from tasks.classification.model import SceneClassificationNetwork

    return SceneClassificationNetwork(model_yaml, classes, weights, device)


def _build_transform() -> Any:
    from tasks.classification.data import SceneDataModule

    return SceneDataModule.build_transform(640, train=False)


class NativeClassification:
    """Load the original strict Head and evaluate with its 640-pixel transform."""

    def __init__(
        self,
        checkpoint: str | Path,
        pretrained_weights: str | Path,
        model_yaml: str | Path,
        device: str | torch.device = "auto",
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.pretrained_weights = Path(pretrained_weights).expanduser().resolve()
        self.model_yaml = Path(model_yaml).expanduser().resolve()
        # Ultralytics resolves scaled YAML aliases (yolov12m-*) to the unified
        # YAML file while retaining scale=m. The alias itself is not a file.
        for path in (self.checkpoint, self.pretrained_weights):
            if not path.is_file():
                raise FileNotFoundError(f"Required classification file is missing: {path}")
        requested = str(device).strip().lower()
        if requested in {"", "auto"}:
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Classification checkpoint must contain a dictionary")
        classes = payload.get("classes")
        if not isinstance(classes, (list, tuple)) or not classes:
            raise ValueError("Classification checkpoint has no class order")
        if any(not isinstance(name, str) or not name.strip() for name in classes):
            raise ValueError("Classification checkpoint has invalid class names")
        self.class_names = list(classes)
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Classification checkpoint has duplicate class names")
        if int(payload.get("num_classes", len(classes))) != len(classes):
            raise ValueError("Classification checkpoint class count differs from its class order")
        self.network = _create_network(
            str(self.model_yaml), len(classes), str(self.pretrained_weights), str(self.device)
        )
        self.network.load_head_checkpoint(self.checkpoint, strict=True)
        self.network.eval()
        self.transform = _build_transform()

    def predict(self, image: Image.Image | str | Path, topk: int = 5) -> dict[str, Any]:
        topk = int(topk)
        if topk <= 0:
            raise ValueError("topk must be positive")
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
        else:
            with Image.open(image) as source:
                source.load()
                rgb = source.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.network(tensor)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, len(self.class_names)):
                raise ValueError("Classification logits differ from the checkpoint class order")
            if not torch.isfinite(logits).all():
                raise ValueError("Classification logits are not finite")
            probabilities = logits.softmax(dim=1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        values, indices = probabilities.topk(min(topk, len(self.class_names)))
        return {
            "predictions": [
                {"class_id": int(index), "class_name": self.class_names[int(index)], "probability": float(value)}
                for value, index in zip(values.detach().cpu(), indices.detach().cpu())
            ],
            "original_size": [rgb.height, rgb.width],
            "imgsz": 640,
            "latency_ms": float(latency_ms),
        }
