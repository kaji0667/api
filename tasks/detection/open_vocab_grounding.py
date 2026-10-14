"""Independent, opt-in GroundingDINO proposals for public localization trials.

Target and reference token spans are scored separately in one caption. No gold,
API requests, HFSA models, or production backend selection enter this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .reference_plan import ReferencePlan, _absolute_position, parse_reference_plan

MODEL_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class QueryCaption:
    caption: str
    spans: tuple[tuple[int, int], ...]
    plan: ReferencePlan
    parse_status: str


def query_caption(description: str) -> QueryCaption:
    if not isinstance(description, str) or not description.strip():
        raise ValueError("A nonempty target description is required")
    if len(description) > 10000:
        raise ValueError("Description exceeds explicit input budget")
    parsed = parse_reference_plan(description)
    if parsed is None:
        valid, position = _absolute_position(description)
        parsed = ReferencePlan(description, description, (), position if valid else None)
        status = "complete_description"
    else:
        status = "target_reference_spans"
    texts = [parsed.target_description, *(ref.description for ref in parsed.references)]
    caption, spans = "", []
    for text in texts:
        phrase = text.strip().rstrip(".。").lower()
        start = len(caption)
        caption += phrase
        spans.append((start, len(caption)))
        caption += ". "
    return QueryCaption(caption.rstrip(), tuple(spans), parsed, status)


def token_spans(offsets: list[tuple[int, int]], spans: tuple[tuple[int, int], ...]) -> list[list[int]]:
    result = [[i for i, (start, end) in enumerate(offsets)
               if end > start and start < right and end > left]
              for left, right in spans]
    if any(not indices for indices in result):
        raise ValueError("Target/reference token span is empty")
    return result


def restore_boxes(boxes_cxcywh: np.ndarray, view_xyxy: tuple[int, int, int, int],
                  image_hw: tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(boxes_cxcywh, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Invalid model proposal tensor")
    left, top, right, bottom = view_xyxy
    center, size = boxes[:, :2], boxes[:, 2:]
    restored = np.concatenate((center - size / 2, center + size / 2), axis=1)
    restored *= np.array([right-left, bottom-top, right-left, bottom-top])
    restored += np.array([left, top, left, top])
    height, width = image_hw
    return np.clip(restored, 0, [width, height, width, height])


def box_iou(a: list[float], b: list[float]) -> float:
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def geometry_score(box: list[float], plan: ReferencePlan, references: list[dict],
                   image_hw: tuple[int, int]) -> float:
    height, width = image_hw
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    terms = []
    if plan.absolute_position is not None:
        for point, extent, direction in zip((cx, cy), (width, height), plan.absolute_position):
            ratio = point / extent
            if direction is not None:
                terms.append(1-ratio if direction == -1 else ratio if direction == 1 else 1-2*abs(ratio-.5))
    for ref, prediction in zip(plan.references, references):
        candidates = prediction["candidates"]
        if not candidates:
            terms.append(.5)
            continue
        # The reference is a model prediction, never an annotation-derived box.
        anchor = max(candidates, key=lambda item: item["score"])["box"]
        ax, ay = (anchor[0]+anchor[2])/2, (anchor[1]+anchor[3])/2
        delta = {"left_of": (ax-cx)/width, "right_of": (cx-ax)/width,
                 "above": (ay-cy)/height, "below": (cy-ay)/height}.get(ref.relation)
        if delta is not None:
            terms.append(float(np.clip(.5+delta, 0, 1)))
        else:
            dx, dy = max(anchor[0]-cx, 0., cx-anchor[2]), max(anchor[1]-cy, 0., cy-anchor[3])
            terms.append(1-min(1., math.hypot(dx, dy)/(.25*math.hypot(width, height))))
    return sum(terms)/len(terms) if terms else 0.


class OpenVocabularyGrounder:
    """Strict local safetensors loading; GPU/CPU device is explicit."""
    def __init__(self, model_dir: str | Path, *, device: str = "cpu", threshold: float = .25,
                 spatial_weight: float = .15, tiled: bool = True) -> None:
        if (isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold < 1
                or isinstance(spatial_weight, bool) or not math.isfinite(spatial_weight)
                or not 0 <= spatial_weight <= 1 or type(tiled) is not bool):
            raise ValueError("Invalid independent grounding configuration")
        import torch
        from transformers import AutoProcessor, GroundingDinoConfig, GroundingDinoForObjectDetection
        root = Path(model_dir).resolve()
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        if provenance.get("revision") != REVISION or provenance["file_sha256"].get("model.safetensors") != MODEL_SHA256:
            raise ValueError("Independent model provenance mismatch")
        for relative, expected in provenance["file_sha256"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or file_sha256(path) != expected:
                raise ValueError("Independent model file hash mismatch")
        config = GroundingDinoConfig.from_pretrained(root, local_files_only=True)
        config.disable_custom_kernels = True  # Official pure PyTorch path; no CUDA extension download/build.
        self.processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        self.model, loading = GroundingDinoForObjectDetection.from_pretrained(
            root, config=config, local_files_only=True, use_safetensors=True,
            output_loading_info=True, torch_dtype=torch.float32,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Independent model requires exact state loading")
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.device, self.threshold, self.spatial_weight, self.tiled = device, threshold, spatial_weight, tiled
        self.max_text_len = config.max_text_len

    def predict(self, image: Image.Image, description: str) -> dict[str, Any]:
        import torch
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("Independent grounder requires an RGB PIL image")
        query = query_caption(description)
        encoded = self.processor.tokenizer(query.caption, return_offsets_mapping=True, truncation=False)
        if len(encoded["input_ids"]) > self.max_text_len:
            raise ValueError("Description exceeds GroundingDINO token budget; no truncation permitted")
        indices = token_spans(encoded["offset_mapping"], query.spans)
        width, height = image.size
        views = [(0, 0, width, height)]
        if self.tiled and max(width, height) > 1600:
            # Bounded four overlapping views preserve detail without an unbounded tile sweep.
            half_w, half_h = math.ceil(width*.6), math.ceil(height*.6)
            views += [(x, y, x+half_w, y+half_h)
                      for y in (0, height-half_h) for x in (0, width-half_w)]
        groups: list[list[dict]] = [[] for _ in indices]
        for view_index, view in enumerate(views):
            inputs = self.processor(images=image.crop(view), text=query.caption, return_tensors="pt")
            if inputs["input_ids"][0].tolist() != encoded["input_ids"]:
                raise ValueError("Processor changed/truncated the registered query tokens")
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                output = self.model(**inputs)
            active_tokens = sorted({position for group in indices for position in group})
            # Official logits use -inf for padded/special text positions. Those
            # positions are deliberately absent from target/reference scores.
            if (not torch.isfinite(output.logits[..., active_tokens]).all()
                    or not torch.isfinite(output.pred_boxes).all()):
                raise ValueError("Nonfinite independent model output")
            probabilities = output.logits[0].sigmoid().float().cpu().numpy()
            boxes = restore_boxes(output.pred_boxes[0].float().cpu().numpy(), view, (height, width))
            for group, positions in zip(groups, indices):
                scores = probabilities[:, positions].max(axis=1)
                for proposal in np.flatnonzero(scores >= self.threshold):
                    box = boxes[proposal].tolist()
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    # Internal crop-boundary fragments cannot displace a complete full-view proposal.
                    clipped = view_index > 0 and any((edge not in (0, extent) and abs(point-edge) <= 2)
                        for point, edge, extent in zip(box, view, (width, height, width, height)))
                    group.append({"box": box, "score": float(scores[proposal]), "view": view_index,
                                  "query_index": int(proposal), "crop_fragment": clipped})
        for group in groups:
            ordered = sorted(group, key=lambda item: (-item["score"], item["view"], item["query_index"]))
            retained = []
            for item in ordered:
                if item["crop_fragment"] and any(other["view"] == 0 and box_iou(item["box"], other["box"]) >= .5 for other in group):
                    continue
                if not any(box_iou(item["box"], other["box"]) >= .7 for other in retained):
                    retained.append(item)
            group[:] = retained
        targets = groups[0]
        refs = [{"description": spec.description, "relation": spec.relation, "candidates": group}
                for spec, group in zip(query.plan.references, groups[1:])]
        for target in targets:
            target["geometry"] = geometry_score(target["box"], query.plan, refs, (height, width))
            target["rank_score"] = target["score"] + self.spatial_weight*target["geometry"]
        raw_index = max(range(len(targets)), key=lambda i: targets[i]["score"]) if targets else None
        selected = max(range(len(targets)), key=lambda i: targets[i]["rank_score"]) if targets else None
        return {"image_hw": [height, width], "parse_status": query.parse_status, "caption": query.caption,
                "target_token_indices": indices[0], "views": views, "candidates": targets, "references": refs,
                "raw_selected_index": raw_index, "selected_index": selected,
                "box": targets[selected]["box"] if selected is not None else None,
                "raw_box": targets[raw_index]["box"] if raw_index is not None else None}
