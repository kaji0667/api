"""Official model entry point; real mode delegates to existing HFSA predictors."""

import os
from pathlib import Path
from typing import Any


class DemoModel:
    """Protocol-only placeholder: it does not inspect images or answer questions."""

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> str | list[int]:
        constraint = request["response_constraint"]
        kind = constraint["type"]
        if kind == "bbox":
            width, height = request["image_width"], request["image_height"]
            return [0, 0, width, height]
        if kind in {"enum", "single_choice"}:
            return constraint["values"][0]
        if kind == "integer":
            return "0"
        return "demo"


class RealModel:
    def __init__(self) -> None:
        from hfsa_adapter import HFSAAdapter

        self.adapter = HFSAAdapter.from_local()

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> str | list[int]:
        return self.adapter.predict(request, image_paths)


def build_model() -> Any:
    mode = os.environ.get("MODEL_MODE", "demo")
    if mode == "demo":
        return DemoModel()
    if mode == "real":
        return RealModel()
    if mode == "qwen":
        from qwen_adapter import QwenModelAdapter
        return QwenModelAdapter()
    raise ValueError("MODEL_MODE must be demo, qwen, or real")
