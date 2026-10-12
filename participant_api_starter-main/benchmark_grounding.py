"""Offline paired diagnostics on public RRSIS-D validation metadata.

No router, HTTP listener, official requests, training or candidate tuning. Gold
boxes/masks are used only after inference for metrics and diagnostic overlays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from api_task_text import TaskInputError
from dataset.rrsisd_refseg_dataset import decode_segmentation_mask
from hfsa_adapter import AdapterFailure, HFSAAdapter, QuestionPlan
from native_detection import NativeDetection, NativeTextEncoder
from native_parity import _capture_forward, _close, _detection_reference, _original_detection_image
from native_refseg import NativeRefSegPredictor
from tasks.refseg.runtime import predict_with_optional_text, resolve_existing_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("split") != split or "segmentation" not in row for row in rows):
        raise ValueError(f"Expected public RRSIS-D {split} rows")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate metadata IDs")
    return rows


def select_rows(rows: list[dict[str, Any]], per_class: int, seed: int) -> list[dict[str, Any]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    rng = random.Random(seed)
    selected = []
    for category in sorted({int(row["class_idx"]) for row in rows}):
        pool = sorted((row for row in rows if int(row["class_idx"]) == category), key=lambda row: str(row["id"]))
        if len(pool) < per_class:
            raise ValueError(f"Not enough validation rows for class {category}")
        selected.extend(sorted(rng.sample(pool, per_class), key=lambda row: str(row["id"])))
    return selected


def box_iou(box: Any, target: Any) -> float:
    if box is None:
        return 0.0
    first, second = np.asarray(box, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,) or not np.isfinite([first, second]).all():
        raise ValueError("IoU expects finite xyxy boxes")
    intersection = np.maximum(np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]), 0).prod()
    area1 = np.maximum(first[2:] - first[:2], 0).prod()
    area2 = np.maximum(second[2:] - second[:2], 0).prod()
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


def mask_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def outward_box(box: Any) -> list[int] | None:
    value = np.asarray(box, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Expected a finite native xyxy candidate")
    if value[2] <= value[0] or value[3] <= value[1]:
        return None
    return [int(np.floor(value[0])), int(np.floor(value[1])), int(np.ceil(value[2])), int(np.ceil(value[3]))]


def aggregate(rows: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    values = [row[backend] for row in rows]
    ious = np.array([value["iou"] for value in values], dtype=np.float64)
    result = {
        "sample_count": len(values), "valid_count": sum(value["box"] is not None for value in values),
        "mean_bbox_iou": float(ious.mean()), "acc_at_0.5": float((ious >= .5).mean()),
        "acc_at_0.7": float((ious >= .7).mean()),
        "mean_bbox_iou_against_mask_box": float(np.mean([value["mask_box_iou"] for value in values])),
        "latency_mean_ms": float(np.mean([value["latency_ms"] for value in values])),
        "latency_max_ms": float(max(value["latency_ms"] for value in values)),
    }
    if backend == "detection":
        oracle = np.array([value["oracle_iou"] for value in values])
        result.update(candidate_recall_at_0_5=float((oracle >= .5).mean()),
                      candidate_recall_at_0_7=float((oracle >= .7).mean()),
                      mean_oracle_iou=float(oracle.mean()),
                      selection_misses_at_0_5=int(((oracle >= .5) & (ious < .5)).sum()),
                      candidate_misses_at_0_5=int((oracle < .5).sum()))
    else:
        result["mean_mask_iou_native"] = float(np.mean([value["mask_iou"] for value in values]))
    return result


def api_box(backend: str, prediction: Any, request: dict[str, Any], image: Path) -> tuple[list[int] | None, str | None]:
    # Reuse the actual adapter's fixed-task path with the already computed real
    # result. This isolates output mapping without repeating model inference.
    predictor = SimpleNamespace(predict=lambda *args, **kwargs: prediction)
    adapter = HFSAAdapter(None, predictor if backend == "detection" else None, None,
                          predictor if backend == "refseg" else None, grounding_backend=backend)
    try:
        return adapter._predict_plan(request, image, QuestionPlan("visual_grounding", request["question"])), None
    except AdapterFailure as error:
        return None, error.code


def detection_parity(native: Any, image: Path, description: str, prediction: Any, observed: dict[str, Any]) -> dict[str, Any]:
    original_hw = prediction.original_size
    tensor, ratio_pad, _ = _original_detection_image(image, int(native.backend.stride), native.image_size_for(original_hw))
    reference = _detection_reference(native, tensor, description, ratio_pad, original_hw, match=True)
    # Independently invert resize/padding, not the legacy scalar-gain helper.
    network_boxes = reference["network_boxes"].numpy().astype(np.float64, copy=True)
    (gain_h, gain_w), (pad_x, pad_y) = ratio_pad
    expected = (network_boxes - np.array([pad_x, pad_y, pad_x, pad_y])) / np.array([gain_w, gain_h, gain_w, gain_h])
    expected = np.clip(expected, 0, [original_hw[1], original_hw[0], original_hw[1], original_hw[0]])
    if prediction.grounding_index != reference["selected"]:
        raise AssertionError("Original/native assignment mismatch")
    if not observed["phrase_maps"] or len(observed["phrase_maps"]) != len(reference["maps"]):
        raise AssertionError("Original/native phrase-map level mismatch")
    return {
        "input_max_abs_error": _close("input", observed["image"], tensor),
        "raw_max_abs_error": _close("raw", observed["raw"], reference["raw"], atol=2e-3),
        "text_max_abs_error": _close("text", observed["conditions"]["txt_vec"], reference["conditions"]["txt_vec"]),
        "phrase_maps_max_abs_error": max((_close("maps", a, b) for a, b in zip(observed["phrase_maps"], reference["maps"])), default=0.0),
        "restored_boxes_max_abs_error": _close("boxes", prediction.boxes, expected, atol=2e-3),
        "scores_max_abs_error": _close("scores", prediction.scores, reference["scores"]),
        "assignment_equal": True,
        "legacy_restored_boxes_max_abs_difference": float(np.max(np.abs(reference["boxes"].numpy() - expected))) if expected.size else 0.0,
    }


def refseg_parity(native: Any, image: Path, description: str, prediction: Any) -> dict[str, Any]:
    from dataset.rrsisd_refseg_dataset import RRSISDRefSegDataset

    rgb = RRSISDRefSegDataset._read_image(str(image))
    resized, _ = RRSISDRefSegDataset._resize(rgb, np.zeros(rgb.shape[:2], dtype=np.uint8), native.imgsz)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255
    features, token_mask = native.text_encoder.encode([description], batch_size=1, return_tokens=True)
    with torch.inference_mode():
        output = predict_with_optional_text(native.model, {"img": tensor, "text_embedding": features, "text_token_mask": token_mask}, True)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        probability = F.interpolate(logits.sigmoid(), size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    error = _close("refseg_original_probability", prediction.probability, probability)
    if not np.array_equal(prediction.mask, probability.numpy() > native.threshold):
        raise AssertionError("Original/native mask mismatch")
    return {"probability_max_abs_error": error, "threshold_mask_equal": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    rows = load_rows(args.annotations, "val")
    train = load_rows(args.train_annotations, "train")
    if {row["image_id"] for row in rows} & {row["image_id"] for row in train}:
        raise ValueError("RRSIS train/val image ID overlap")
    selected = select_rows(rows, args.per_class, args.seed)
    assets = {
        "detection": args.models_dir / "runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt",
        "refseg": args.models_dir / "runs/semseg/srp_yolov12m_axis/weights/best_raw.pt",
        "clip": args.models_dir / "openclip/ViT-L-14.pt",
    }
    for path in assets.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_paths = [Path(__file__), Path(__file__).with_name("native_detection.py"), Path(__file__).with_name("native_refseg.py"),
                    Path(__file__).with_name("native_parity.py"), Path(__file__).with_name("hfsa_adapter.py"),
                    CODE_DIR / "tasks/detection/test.py", CODE_DIR / "ultralytics/utils/ops.py",
                    CODE_DIR / "ultralytics/nn/modules/text_backbone.py"]
    hashes_before = {str(path): sha256(path) for path in [*assets.values(), *source_paths]}
    started = time.perf_counter()
    print("Loading existing CPU CLIP and detection weights", flush=True)
    encoder = NativeTextEncoder(device="cpu", pretrained=str(assets["clip"]))
    detection = NativeDetection(assets["detection"], device="cpu", encoder=encoder, max_image_size=1600)

    def existing_encoder(**kwargs: Any) -> Any:
        if kwargs["model_name"] != encoder.encoder.model_name or kwargs["device"] != "cpu" or kwargs["precision"] != "fp32":
            raise ValueError("RefSeg text encoder configuration mismatch")
        return encoder.encoder

    print("Loading existing CPU RefSeg weights with the same CLIP", flush=True)
    refseg = NativeRefSegPredictor(assets["refseg"], device="cpu", text_encoder_factory=existing_encoder)
    load_ms = (time.perf_counter() - started) * 1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parity_indices = set(np.linspace(0, len(selected) - 1, min(args.parity_samples, len(selected)), dtype=int).tolist())
    details = []
    for index, row in enumerate(selected):
        raw_path = Path(row["image"])
        image = resolve_existing_path(raw_path) if raw_path.is_absolute() else resolve_existing_path(args.data_root / raw_path)
        if image is None:
            raise FileNotFoundError(raw_path)
        decoded = cv2.imread(str(image))
        if decoded is None:
            raise ValueError(f"Cannot decode public image {row['id']}")
        height, width = decoded.shape[:2]
        target = row["bbox"]
        if len(target) != 4 or not (0 <= target[0] < target[2] <= width and 0 <= target[1] < target[3] <= height):
            raise ValueError("Invalid public annotation xyxy bbox")
        gold_mask = decode_segmentation_mask(row["segmentation"]).astype(bool)
        gold_box = mask_box(gold_mask)
        if gold_mask.shape != (height, width) or gold_box is None:
            raise ValueError("Invalid public reference mask")
        question = "Return the bounding box. Description: " + row["text"]
        request = {"question": question, "image_height": height, "image_width": width, "response_constraint": {"type": "bbox"}}
        detail = {"id": row["id"], "class_idx": row["class_idx"], "class_name": row["class_name"],
                  "image": str(image), "image_sha256": sha256(image), "original_hw": [height, width], "text": row["text"],
                  "annotation_bbox_xyxy": target, "mask_bbox_xyxy_exclusive": gold_box,
                  "annotation_mask_bbox_iou": box_iou(target, gold_box)}
        observed = {}
        for backend, native in (("detection", detection), ("refseg", refseg)):
            started = time.perf_counter()
            prediction = None
            try:
                if backend == "detection" and index in parity_indices:
                    with _capture_forward(detection.model) as observed:
                        prediction = native.predict(image, question)
                else:
                    prediction = native.predict(image, question)
                latency = (time.perf_counter() - started) * 1000
                box, error = api_box(backend, prediction, request, image)
            except TaskInputError as failure:
                latency = (time.perf_counter() - started) * 1000
                box, error = None, failure.code
            value = {"box": box, "error": error, "iou": box_iou(box, target), "mask_box_iou": box_iou(box, gold_box), "latency_ms": latency}
            if backend == "detection":
                candidates = prediction.boxes.tolist() if prediction is not None else []
                api_candidates = [outward_box(candidate) for candidate in candidates]
                value.update(candidate_count=len(candidates), candidates_xyxy=candidates,
                             candidate_boxes_api_xyxy=api_candidates,
                             scores=prediction.scores.tolist() if prediction is not None else [],
                             selected_index=prediction.grounding_index if prediction is not None else None,
                             oracle_iou=max((box_iou(candidate, target) for candidate in api_candidates), default=0.0))
                if prediction is not None and index in parity_indices:
                    value["original_parity"] = detection_parity(native, image, row["text"], prediction, observed)
            else:
                intersection = np.count_nonzero(prediction.mask & gold_mask) if prediction is not None else 0
                union = np.count_nonzero(prediction.mask | gold_mask) if prediction is not None else np.count_nonzero(gold_mask)
                value["mask_iou"] = intersection / union
                if prediction is not None and index in parity_indices:
                    value["original_parity"] = refseg_parity(native, image, row["text"], prediction)
            detail[backend] = value
        details.append(detail)
        if index < args.overlays:
            preview = decoded.copy()
            for label, box, color in (("GT", target, (0, 255, 0)), ("DET", detail["detection"]["box"], (0, 0, 255)), ("SEG", detail["refseg"]["box"], (255, 0, 0))):
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(preview, label, (x1, max(18, y1)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            preview_path = args.output.parent / f"public_{row['id']}.jpg"
            if not cv2.imwrite(str(preview_path), preview):
                raise OSError("Could not write public overlay")
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(selected):
            print(json.dumps({"progress": index + 1, "total": len(selected)}), flush=True)
    if hashes_before != {path: sha256(Path(path)) for path in hashes_before}:
        raise RuntimeError("Source or model asset changed during diagnostic")
    return {"protocol": "public_rrsisd_val_stratified_bbox_diagnostic_v1", "seed": args.seed, "per_class": args.per_class,
            "device": "cpu", "threads": args.threads, "torch_version": torch.__version__, "model_load_ms": load_ms,
            "annotations": str(args.annotations), "annotations_sha256": sha256(args.annotations),
            "train_annotations_sha256": sha256(args.train_annotations), "rrsis_train_val_image_id_overlap": 0,
            "detection_training_image_overlap": "unknown; original detection training split/cache not available",
            "geographic_source_isolation": "unknown", "source_and_asset_sha256": hashes_before,
            "settings": {"detection_cap": 1600, "detection_conf": .25, "nms_iou": .7, "max_det": 300,
                         "matching_temperature": .7, "refseg_imgsz": refseg.imgsz, "refseg_threshold": refseg.threshold},
            "metric_target": "annotation bbox interpreted as xyxy; mask-derived exclusive bbox reported separately",
            "oracle_metric": "all native candidates rounded outward by the same API box rule; diagnostic only",
            "api_validation": "fixed-task native-result to actual adapter output mapping; no HTTP or Ollama routing",
            "parity_sample_count": len(parity_indices), "gold_used_for_inference": False,
            "summary": {backend: aggregate(details, backend) for backend in ("detection", "refseg")},
            "by_class": {str(category): {backend: aggregate([row for row in details if row["class_idx"] == category], backend)
                                       for backend in ("detection", "refseg")} for category in sorted({row["class_idx"] for row in details})},
            "annotation_mask_bbox_mean_iou": float(np.mean([row["annotation_mask_bbox_iou"] for row in details])), "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parity-samples", type=int, default=4)
    parser.add_argument("--overlays", type=int, default=8)
    args = parser.parse_args()
    if args.threads <= 0 or args.parity_samples < 0 or args.overlays < 0:
        parser.error("threads must be positive; parity-samples and overlays must be nonnegative")
    report = run(args)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "summary": report["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
