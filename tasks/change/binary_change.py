"""Independent, strictly pretrained DSIFN binary change evidence.

This branch predicts generic binary change rather than MCI's road/building
classes. It supplies neither event direction nor semantic object categories.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ARCHITECTURE_ID = "changeformer_v6_dsifn_binary_v1"
SOURCE_REVISION = "5ca45352d06e15a01e2d6525d891f9c5bbf54f52"
CHECKPOINT_SHA256 = "e983f39209952cf7b3e341b54f4ff48115fcba979542d61bc5193c270396acfc"
NETWORK_SHA256 = "462595fc8dbd0b7ea023ebe6a5099f57659c15fef7e2d09edccbc1b60c69966b"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_binary_pair(before: str | Path, after: str | Path
                     ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Use the author's eval RGB/PIL bicubic/ToTensor/0.5 normalization."""
    with Image.open(before) as image:
        first = image.convert("RGB")
    with Image.open(after) as image:
        second = image.convert("RGB")
    if first.size != second.size:
        raise ValueError("image_dimensions_mismatch")
    original_hw = (first.height, first.width)
    if first.size != (256, 256):
        first = TF.resize(first, [256, 256], interpolation=InterpolationMode.BICUBIC)
        second = TF.resize(second, [256, 256], interpolation=InterpolationMode.BICUBIC)
    tensors = [TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)
               .unsqueeze(0).contiguous() for image in (first, second)]
    return tensors[0], tensors[1], original_hw


@dataclass(frozen=True)
class BinaryChangeEvidence:
    mask: np.ndarray
    elapsed_seconds: float
    original_hw: tuple[int, int]

    def summary(self) -> dict:
        from .regions import mask_region_pixels

        if (not isinstance(self.mask, np.ndarray) or self.mask.ndim != 2
                or self.mask.dtype != np.uint8
                or not set(np.unique(self.mask)).issubset({0, 1})):
            raise ValueError("Invalid binary change mask")
        return {"mask_scope": "generic_binary_change",
                "mask_grid": {"binary": {"changed_pixels": int(self.mask.sum()),
                                         "region_pixels": mask_region_pixels(self.mask)}},
                "mask_direction": "not available",
                "image_order": "A before, B after",
                "mask_size": list(self.mask.shape)}


class BinaryChangePredictor:
    """A persistent model instance; no checkpoint compatibility fallbacks."""

    def __init__(self, bundle_dir: str | Path, device: str = "cpu") -> None:
        root = Path(bundle_dir).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("architecture_id") != ARCHITECTURE_ID
                or manifest.get("source_revision") != SOURCE_REVISION
                or manifest.get("vendored_network_sha256") != NETWORK_SHA256
                or file_sha256(Path(__file__).parent / "_binary/network.py") != NETWORK_SHA256
                or manifest.get("preprocessing") != "author_eval_rgb_pil_bicubic_256_to_tensor_norm_0.5"
                or manifest.get("model") != {"input_nc": 3, "output_nc": 2,
                                            "embed_dim": 256, "decoder_softmax": False}):
            raise ValueError("Unexpected binary change model bundle")
        item = manifest["checkpoint"]
        checkpoint_path = (root / item["path"]).resolve()
        if (not checkpoint_path.is_relative_to(root) or not checkpoint_path.is_file()
                or item.get("sha256") != CHECKPOINT_SHA256
                or checkpoint_path.stat().st_size != item.get("size")
                or file_sha256(checkpoint_path) != CHECKPOINT_SHA256):
            raise ValueError("Binary change checkpoint integrity mismatch")
        from ._binary.network import ChangeFormerV6

        self.device = torch.device(device)
        self.checkpoint_sha256 = CHECKPOINT_SHA256
        self.bundle_dir = root
        # The author constructor initializes parameters; strict loading replaces
        # every parameter/buffer and preserves the caller's CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            model = ChangeFormerV6(input_nc=3, output_nc=2, embed_dim=256,
                                   decoder_softmax=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected = model.state_dict()
        if (not isinstance(state, dict) or set(state) != set(expected)
                or any(not isinstance(value, torch.Tensor) for value in state.values())):
            raise ValueError("Binary change checkpoint key mismatch")
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError("Binary change checkpoint shape/dtype mismatch")
            if not torch.isfinite(value).all():
                raise ValueError("Binary change checkpoint has nonfinite tensors")
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False).eval().to(self.device)
        self.model = model
        self.loading = {"checkpoint_tensors": len(state), "matched_tensors": len(expected),
                        "missing": 0, "unexpected": 0, "strict_state": True,
                        "weights_only": True,
                        "parameters": sum(parameter.numel() for parameter in model.parameters())}

    @torch.inference_mode()
    def evidence(self, before: str | Path, after: str | Path) -> BinaryChangeEvidence:
        started = time.monotonic()
        first, second, original_hw = read_binary_pair(before, after)
        outputs = self.model(first.to(self.device), second.to(self.device))
        if not isinstance(outputs, list) or len(outputs) != 5:
            raise ValueError("Invalid binary change prediction outputs")
        logits = outputs[-1]
        if logits.shape != (1, 2, 256, 256) or not torch.isfinite(logits).all():
            raise ValueError("Invalid binary change logits")
        mask = logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)
        return BinaryChangeEvidence(mask, time.monotonic() - started, original_hw)
