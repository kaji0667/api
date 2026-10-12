"""Checkpoint-owning single-image inference for scene classification."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from .data import SceneDataModule
from .model import SceneClassificationNetwork


NetworkFactory = Callable[..., Any]


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = str(device).strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


class SceneClassificationPredictor:
    """Load one classification checkpoint and serve repeated Top-K predictions."""

    def __init__(
        self,
        checkpoint: str | Path,
        pretrained_weights: str | Path = "pretrain_model/yolov12m.pt",
        device: str | torch.device = "auto",
        imgsz: int = 640,
        model_yaml: str = "",
        network_factory: NetworkFactory = SceneClassificationNetwork,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.pretrained_weights = Path(pretrained_weights).expanduser().resolve()
        self.device = _resolve_device(device)
        self.imgsz = int(imgsz)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Scene classification checkpoint not found: {self.checkpoint}")
        if not self.pretrained_weights.is_file():
            raise FileNotFoundError(f"YOLOv12 pretrained weights not found: {self.pretrained_weights}")
        if self.imgsz <= 0:
            raise ValueError("imgsz must be positive.")

        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Scene classification checkpoint must contain a dictionary payload.")
        self.class_names = [str(name) for name in payload.get("classes", [])]
        if not self.class_names:
            raise ValueError("Checkpoint does not contain the scene class list.")
        checkpoint_num_classes = int(payload.get("num_classes", len(self.class_names)))
        if checkpoint_num_classes != len(self.class_names):
            raise ValueError("Checkpoint num_classes does not match its class list.")

        resolved_yaml = str(model_yaml).strip() or str(
            payload.get("model_yaml", "ultralytics/cfg/models/v12/yolov12m-classification.yaml")
        )
        self.network = network_factory(
            resolved_yaml,
            len(self.class_names),
            str(self.pretrained_weights),
            str(self.device),
        )
        self.network.load_head_checkpoint(self.checkpoint, strict=True)
        self.network.eval()
        self.transform = SceneDataModule.build_transform(self.imgsz, train=False)

    @staticmethod
    def _load_image(image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(image) as source:
            source.load()
            return source.convert("RGB")

    def predict(self, image: Image.Image | str | Path, topk: int = 5) -> dict[str, Any]:
        requested_topk = int(topk)
        if requested_topk <= 0:
            raise ValueError("topk must be positive.")
        rgb_image = self._load_image(image)
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            probabilities = self.network(tensor).softmax(dim=1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        values, indices = probabilities.topk(min(requested_topk, len(self.class_names)))
        predictions = [
            {
                "class_id": int(index),
                "class_name": self.class_names[int(index)],
                "probability": float(value),
            }
            for value, index in zip(values.detach().cpu(), indices.detach().cpu())
        ]
        return {
            "predictions": predictions,
            "latency_ms": float(latency_ms),
            "original_size": [int(rgb_image.height), int(rgb_image.width)],
            "imgsz": self.imgsz,
        }
