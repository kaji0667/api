"""Strict, persistent CPU-capable inference for the author's pretrained MCI.

Images must depict the same registered footprint, in before/after order.
The caption branch describes changes; the mask branch labels only roads and
buildings. It does not supply arbitrary land-cover transitions or object counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

ARCHITECTURE_ID = "change_agent_mci_v1"
CHECKPOINT_SHA256 = "34c6926342c40fdd6d50b43d50257cb46cb6b61763448de9ee551828a64b3eb9"
MEAN = np.asarray([0.39073 * 255, 0.38623 * 255, 0.32989 * 255], dtype=np.float32)
STD = np.asarray([0.15329 * 255, 0.14628 * 255, 0.13648 * 255], dtype=np.float32)
REASONS = frozenset({"unsupported_task", "unsupported_constraint", "insufficient_evidence",
                     "ambiguous_region", "invalid_input", "image_dimensions_mismatch",
                     "invalid_options", "input_too_long", "answer_unmatched",
                     "answerer_unavailable_or_timeout", "answerer_invalid_output"})


@dataclass(frozen=True)
class ChangeResult:
    supported: bool
    answer: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ChangeEvidence:
    caption: str
    mask: np.ndarray
    elapsed_seconds: float
    original_hw: tuple[int, int]

    def summary(self) -> dict[str, Any]:
        """Model-derived measurements only; never use GT in answer generation."""
        labels = {"road": 1, "building": 2, "all": None}
        grids = {}
        for label, class_id in labels.items():
            foreground = self.mask != 0 if class_id is None else self.mask == class_id
            counts = {}
            # Explicit edges partition every pixel, including non-divisible 256.
            edges = [0, 85, 170, 256]
            names = ("top_left", "top_center", "top_right", "middle_left", "center",
                     "middle_right", "bottom_left", "bottom_center", "bottom_right")
            for index, name in enumerate(names):
                row, col = divmod(index, 3)
                counts[name] = int(foreground[edges[row]:edges[row + 1],
                                              edges[col]:edges[col + 1]].sum())
            counts.update({"upper_left": int(foreground[:128, :128].sum()),
                           "upper_right": int(foreground[:128, 128:].sum()),
                           "lower_left": int(foreground[128:, :128].sum()),
                           "lower_right": int(foreground[128:, 128:].sum())})
            grids[label] = {"changed_pixels": int(foreground.sum()), "region_pixels": counts}
        return {"caption": self.caption, "mask_grid": grids,
                "mask_scope": "road/building change; background is not a general no-change class",
                "mask_direction": "not available; added/removed direction only if explicit in caption",
                "image_order": "A before, B after", "mask_size": [256, 256]}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_file(root: Path, item: dict[str, Any]) -> Path:
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or invalid MCI bundle file")
    if file_sha256(path) != item["sha256"]:
        raise ValueError("MCI bundle file checksum mismatch")
    return path


def read_pair(before: str | Path, after: str | Path) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    arrays = []
    for path in (before, after):
        with Image.open(path) as image:
            # Do not independently rotate registered imagery via EXIF.
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.float32))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("image_dimensions_mismatch")
    original_hw = tuple(arrays[0].shape[:2])
    tensors = []
    for array in arrays:
        if array.shape[:2] != (256, 256):
            array = cv2.resize(array, (256, 256), interpolation=cv2.INTER_LINEAR)
        # On native 256 RGB this equals the upstream per-channel CHW transform.
        array = ((array - MEAN) / STD).transpose(2, 0, 1)
        tensors.append(torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0))
    return tensors[0], tensors[1], original_hw


def load_pretrained_state(module: torch.nn.Module, state: dict[str, torch.Tensor], *,
                          ignored: dict[str, Any] | None = None) -> dict[str, Any]:
    """Strictly cover current graph, allowing only exact audited author leftovers.

    The fixed released fusion file has 942 keys absent from the author's public
    inference graph, which their predict/test load with strict=False. We check
    every such key's name/shape/dtype; all current graph keys load with strict=True.
    This is not a general checkpoint compatibility fallback.
    """
    ignored = {} if ignored is None else ignored
    required = module.state_dict()
    if (not isinstance(state, dict) or set(required) - set(state)
            or set(state) - set(required) != set(ignored)):
        raise ValueError("MCI checkpoint state keys do not match the audited inference graph")
    for key, tensor in state.items():
        expected = required[key] if key in required else ignored[key]
        shape = list(expected.shape) if isinstance(expected, torch.Tensor) else expected["shape"]
        dtype = str(expected.dtype) if isinstance(expected, torch.Tensor) else expected["dtype"]
        if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != shape or str(tensor.dtype) != dtype):
            raise ValueError("MCI checkpoint state shape/dtype mismatch")
    module.load_state_dict({key: state[key] for key in required}, strict=True)
    return {"checkpoint_tensors": len(state), "matched_tensors": len(required),
            "ignored_inactive_tensors": len(ignored), "missing": 0, "unexpected": 0,
            "strict_active_state": True}


class ChangePredictor:
    def __init__(self, *, bundle_dir: str | Path, device: str = "cpu", answerer: Any = None,
                 binary_predictor: Any = None) -> None:
        root = Path(bundle_dir).expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("architecture_id") != ARCHITECTURE_ID
                or manifest.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
                or manifest.get("preprocessing") != {"size": 256, "interpolation": "opencv_linear",
                                                     "mean": MEAN.tolist(), "std": STD.tolist()}):
            raise ValueError("Unsupported MCI architecture/checkpoint/preprocessing")
        checkpoint_path = bundle_file(root, manifest["checkpoint"])
        vocab_path = bundle_file(root, manifest["vocabulary"])
        inactive_path = bundle_file(root, manifest["checkpoint_inactive_keys"])
        inactive = json.loads(inactive_path.read_text(encoding="utf-8"))
        if (inactive.get("checkpoint_sha256") != CHECKPOINT_SHA256
                or inactive.get("source_revision") != "68cbaa7f388b36e4fc10872f7a2911482d26ae5b"
                or inactive.get("encoder_source_sha256") != "33dba6e33a295f84ed4fb5ec07065dfd36a681969a69d6296380d6244758ddd3"
                or set(inactive.get("ignored_state_dict_keys", {})) != {"encoder_trans_dict"}
                or len(inactive["ignored_state_dict_keys"]["encoder_trans_dict"]) != 942):
            raise ValueError("MCI inactive checkpoint keys are not the audited release")
        self.vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
        if (len(self.vocabulary) != 468 or set(self.vocabulary.values()) != set(range(468))
                or any(token not in self.vocabulary for token in ("<START>", "<END>", "<NULL>"))):
            raise ValueError("MCI vocabulary does not match pretrained decoder")
        self.words = {index: token for token, index in self.vocabulary.items()}
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("MCI supports CPU or CUDA devices")
        from ._mci.model_encoder_att import Encoder, AttentiveEncoder
        from ._mci.model_decoder import DecoderTransformer

        self.encoder = Encoder("segformer-mit_b1")
        self.fusion = AttentiveEncoder(train_stage=None, n_layers=3,
                                       feature_size=[16, 16, 512], heads=8, dropout=0.1)
        self.decoder = DecoderTransformer(encoder_dim=512, feature_dim=512,
                                           vocab_size=468, max_lengths=41,
                                           word_vocab=self.vocabulary, n_head=8,
                                           n_layers=1, dropout=0.1)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        self.loading = {}
        for name, module in (("encoder_dict", self.encoder), ("encoder_trans_dict", self.fusion),
                             ("decoder_dict", self.decoder)):
            state = checkpoint[name]
            self.loading[name] = load_pretrained_state(
                module, state, ignored=inactive["ignored_state_dict_keys"].get(name))
            module.to(self.device).eval().requires_grad_(False)
        del checkpoint
        self.parameter_count = sum(p.numel() for module in (self.encoder, self.fusion, self.decoder)
                                   for p in module.parameters())
        if answerer is None:
            from .answering import EvidenceAnswerer
            answerer = EvidenceAnswerer()
        self.answerer = answerer
        self.checkpoint_sha256 = CHECKPOINT_SHA256
        self.binary_predictor = binary_predictor

    @torch.inference_mode()
    def evidence(self, before: str | Path, after: str | Path) -> ChangeEvidence:
        started = time.monotonic()
        image_a, image_b, original_hw = read_pair(before, after)
        feature_a, feature_b = self.encoder(image_a.to(self.device), image_b.to(self.device))
        feature_a, feature_b, mask_logits = self.fusion(feature_a, feature_b)
        if mask_logits.shape != (1, 3, 256, 256) or not torch.isfinite(mask_logits).all():
            raise ValueError("Invalid MCI segmentation output")
        sequence = self.decoder.sample(feature_a, feature_b, k=1)
        special = {self.vocabulary[token] for token in ("<START>", "<END>", "<NULL>")}
        tokens = [self.words[int(index)] for index in sequence if int(index) not in special]
        caption = " ".join(tokens).strip()
        if not caption or len(caption) > 4096:
            raise ValueError("Invalid MCI caption output")
        mask = mask_logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)
        return ChangeEvidence(caption, mask, time.monotonic() - started, original_hw)

    def predict(self, before: str | Path, after: str | Path, question: str, *,
                response_constraint: dict[str, Any], choices: dict[str, str] | None = None) -> ChangeResult:
        if not isinstance(question, str) or not question.strip():
            return ChangeResult(False, reason="invalid_input")
        if response_constraint.get("type") not in {"short_text", "enum", "single_choice"}:
            return ChangeResult(False, reason="unsupported_constraint")
        from .answering import generic_region_query
        generic_region = generic_region_query(question)
        if generic_region and response_constraint.get("type") not in {"enum", "single_choice"}:
            return ChangeResult(False, reason="unsupported_constraint")
        if generic_region and self.binary_predictor is None:
            return ChangeResult(False, reason="insufficient_evidence")
        try:
            evidence = (self.binary_predictor.evidence(before, after) if generic_region
                        else self.evidence(before, after))
        except (OSError, ValueError) as error:
            reason = "image_dimensions_mismatch" if str(error) == "image_dimensions_mismatch" else "invalid_input"
            return ChangeResult(False, reason=reason)
        return self.answerer.answer(question, response_constraint, choices, evidence.summary())
