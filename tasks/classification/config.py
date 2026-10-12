"""Configuration object for the scene-classification task."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class SceneClassificationConfig:
    data_dir: str = "data/VRSBench_scene"
    model: str = "ultralytics/cfg/models/v12/yolov12m-classification.yaml"
    weights: str = "pretrain_model/yolov12m.pt"
    save_dir: str = "runs/classification/vrsbench_scene"
    batch: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip_norm: float = 1.0
    imgsz: int = 640
    val_ratio: float = 0.2
    seed: int = 42
    workers: int = 2
    device: str = "cuda:0"
    sampling: str = "balanced"
    max_train_batches: int = 0
    max_val_batches: int = 0

    def validate(self) -> None:
        if self.batch <= 0 or self.epochs <= 0 or self.imgsz <= 0:
            raise ValueError("batch, epochs, and imgsz must be positive.")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1.")
        if self.sampling not in {"balanced", "none"}:
            raise ValueError("sampling must be 'balanced' or 'none'.")
        if self.max_train_batches < 0 or self.max_val_batches < 0:
            raise ValueError("max batch limits cannot be negative.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
