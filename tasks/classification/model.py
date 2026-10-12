"""YOLOv12m feature extractor plus the integrated multi-scale scene head."""

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from ultralytics.nn.modules import SceneClassifyHead
from ultralytics.nn.tasks import ClassificationModel


class SceneClassificationNetwork(nn.Module):
    """Build the classification YAML, load matching YOLO weights, and freeze feature layers."""

    def __init__(self, model_yaml, num_classes, pretrained_weights="", device="cpu"):
        super().__init__()
        self.model_yaml = str(model_yaml)
        self.model = ClassificationModel(self.model_yaml, ch=3, nc=int(num_classes), verbose=False)
        self.configure_feature_extractor_batchnorm()
        self.device = torch.device(device)
        self.pretrained_report = {"matched_tensors": 0, "skipped_tensors": 0}
        if pretrained_weights:
            self.pretrained_report = self.load_pretrained(pretrained_weights)
        self.freeze_feature_extractor()
        self.to(self.device)

    @property
    def head(self) -> SceneClassifyHead:
        head = self.model.model[-1]
        if not isinstance(head, SceneClassifyHead):
            raise TypeError(f"Expected SceneClassifyHead, got {type(head).__name__}.")
        return head

    def forward(self, images):
        return self.model(images)

    def configure_feature_extractor_batchnorm(self):
        """Match the BatchNorm settings used by the YOLO DetectionModel that trained the frozen features."""
        for module in self.model.model[:-1].modules():
            if type(module) is nn.BatchNorm2d:
                module.eps = 1e-3
                module.momentum = 0.03

    @staticmethod
    def _state_dict_from_checkpoint(checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if hasattr(checkpoint, "state_dict"):
            checkpoint = checkpoint.state_dict()
        if not isinstance(checkpoint, dict):
            raise TypeError("Unsupported pretrained checkpoint format.")
        return checkpoint

    def load_pretrained(self, weights):
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YOLOv12 pretrained weights not found: {path}")
        source = self._state_dict_from_checkpoint(torch.load(path, map_location="cpu", weights_only=False))
        target = self.model.state_dict()
        matched = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
        self.model.load_state_dict(matched, strict=False)
        return {"matched_tensors": len(matched), "skipped_tensors": len(source) - len(matched), "weights": str(path)}

    def freeze_feature_extractor(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        self.head.train(mode)
        return self

    def load_head_checkpoint(self, checkpoint, strict=True):
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("classify_head", payload.get("head", payload))
        self.head.load_state_dict(state, strict=strict)
        return payload

    def checkpoint_payload(self, classes, epoch, val_metrics, config):
        return {
            "task": "scene_classification",
            "classify_head": self.head.state_dict(),
            "classes": list(classes),
            "num_classes": len(classes),
            "epoch": int(epoch),
            "val_acc": float(val_metrics["top1_accuracy"]),
            "val_metrics": dict(val_metrics),
            "model_yaml": self.model_yaml,
            "pretrained_report": dict(self.pretrained_report),
            "config": dict(config),
        }
