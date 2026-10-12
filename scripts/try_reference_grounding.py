"""Public offline trial: detect references first, then search a target ROI.

Selection depends on public description syntax and IDs only. Annotations enter
metrics after every prediction; this command never starts HTTP or uses official
requests. The injected task pipeline does not depend on the API package.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from tasks.detection.reference_plan import parse_reference_plan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_selection(path: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("split") != "val" for row in rows):
        raise ValueError("Expected public validation metadata")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate public sample IDs")
    eligible = []
    for row in sorted(rows, key=lambda value: str(value["id"])):
        # No box, mask, reference labels or predictions participate in selection.
        plan = parse_reference_plan(row["text"])
        if plan is not None:
            eligible.append({"row": row, "plan": plan})
    if not eligible:
        raise ValueError("No supported reference expressions in public metadata")
    return eligible[:limit], {
        "metadata_count": len(rows), "eligible_count": len(eligible),
        "eligible_multiple_reference_count": sum(len(value["plan"].references) > 1 for value in eligible),
    }


def image_path(row: dict[str, Any], root: Path) -> Path:
    root = root.expanduser().resolve()
    path = (root / row["image"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Public image must exist under the declared data root")
    return path


def outward_box(box: Any) -> list[int] | None:
    if box is None:
        return None
    values = list(map(float, box))
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected finite xyxy coordinates")
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return [math.floor(values[0]), math.floor(values[1]), math.ceil(values[2]), math.ceil(values[3])]


def box_iou(first: Any, second: Any) -> float:
    if first is None:
        return 0.0
    a, b = list(map(float, first)), list(map(float, second))
    if len(a) != 4 or len(b) != 4 or not all(math.isfinite(value) for value in a + b):
        raise ValueError("Expected two finite xyxy boxes")
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def prediction_value(prediction: Any) -> dict[str, Any]:
    boxes = prediction.boxes.tolist()
    index = prediction.grounding_index
    if index is not None and not 0 <= int(index) < len(boxes):
        raise ValueError("Invalid original description matching index")
    return {"box": outward_box(boxes[int(index)]) if index is not None else None,
            "candidate_boxes": boxes, "selected_index": int(index) if index is not None else None}


def scored(value: dict[str, Any], target: Any) -> dict[str, Any]:
    result = dict(value)
    result["iou"] = box_iou(result["box"], target)
    result["oracle_iou"] = max((box_iou(outward_box(box), target) for box in result["candidate_boxes"]), default=0.0)
    return result


def aggregate(details: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [detail[key] for detail in details]
    count = len(values)
    return {"sample_count": count, "valid_count": sum(value["box"] is not None for value in values),
            "correct_at_0_5": sum(value["iou"] >= .5 for value in values),
            "acc_at_0_5": sum(value["iou"] >= .5 for value in values) / count,
            "mean_iou": sum(value["iou"] for value in values) / count,
            "candidate_recall_at_0_5": sum(value["oracle_iou"] >= .5 for value in values) / count,
            "mean_oracle_iou": sum(value["oracle_iou"] for value in values) / count}


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected, coverage = public_selection(args.annotations, args.limit)
    train_rows = [json.loads(line) for line in args.train_annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("Expected public training metadata for split audit")
    if {entry["row"]["image_id"] for entry in selected} & {row["image_id"] for row in train_rows}:
        raise ValueError("Selected images overlap the declared RRSIS training split")
    manifest = [{"id": entry["row"]["id"], "image": str(image_path(entry["row"], args.data_root)),
                 "plan": asdict(entry["plan"])} for entry in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.selection_only and (args.output / "results.jsonl").exists():
        raise ValueError("An existing trial is preserved; choose a new output directory")
    selection_path = args.output / "selection.json"
    selection_content = json.dumps({"selection": "supported syntax, IDs ascending, fixed first N",
                                    "annotations_sha256": sha256(args.annotations),
                                    "coverage": coverage, "samples": manifest}, ensure_ascii=False, indent=2) + "\n"
    if selection_path.exists():
        if selection_path.read_text(encoding="utf-8") != selection_content:
            raise ValueError("Preselected samples changed; preserve this trial and choose another output directory")
    else:
        selection_path.write_text(selection_content, encoding="utf-8")
    print(json.dumps({"selection": str(selection_path), "selected_count": len(selected), **coverage}), flush=True)
    if args.selection_only:
        return {"selection_only": True, "coverage": coverage, "samples": manifest}

    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(CODE_DIR / "participant_api_starter-main"))
    from native_detection import NativeDetection, NativeTextEncoder
    from tasks.detection.reference_grounding import predict_reference_grounding

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    models_root = args.models_dir.expanduser().resolve()
    assets = {"detection": models_root / "runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt",
              "clip": models_root / "openclip/ViT-L-14.pt"}
    protected_sources = [Path(__file__), CODE_DIR / "tasks/detection/reference_plan.py",
                         CODE_DIR / "tasks/detection/reference_grounding.py",
                         CODE_DIR / "participant_api_starter-main/native_detection.py",
                         CODE_DIR / "participant_api_starter-main/grounding_spatial.py",
                         CODE_DIR / "tasks/detection/test.py", CODE_DIR / "text_encoder/matching.py",
                         CODE_DIR / "ultralytics/nn/modules/text_backbone.py", CODE_DIR / "text_encoder/model.py"]
    hashes_before = {str(path): sha256(path) for path in [*assets.values(), *protected_sources]}
    started = time.perf_counter()
    print("Loading original CPU detector and local OpenCLIP for an offline trial", flush=True)
    encoder = NativeTextEncoder(device="cpu", pretrained=str(assets["clip"]))
    detector = NativeDetection(assets["detection"], device="cpu", encoder=encoder,
                               max_image_size=1600, spatial_weight=0)
    model_load_ms = (time.perf_counter() - started) * 1000
    details = []
    progress_path = args.output / "results.jsonl"
    with progress_path.open("w", encoding="utf-8") as output:
        for index, (entry, sample) in enumerate(zip(selected, manifest)):
            row, plan = entry["row"], entry["plan"]
            image = cv2.imread(sample["image"])
            if image is None:
                raise ValueError("Public image decode failed")
            sample_started = time.perf_counter()
            # The actual reference-first pipeline runs before every control arm.
            trial = predict_reference_grounding(detector, image, plan)
            full = trial["full_target"]
            semantic_value = {"box": outward_box(full["selected_box"]),
                              "candidate_boxes": full["candidate_boxes"], "selected_index": full["selected_index"]}
            # Current endpoint policy is a separate full-image control. Its old
            # absolute-position bonus must never use crop-relative coordinates.
            detector.spatial_weight = .15
            try:
                current = detector.predict(image, plan.original_description)
            finally:
                detector.spatial_weight = 0
            current_value = prediction_value(current)
            if not np.array_equal(np.asarray(current_value["candidate_boxes"]), np.asarray(semantic_value["candidate_boxes"])):
                raise AssertionError("Selection bonus changed full-image candidates")
            subject = prediction_value(detector.predict(image, plan.target_description))
            new_candidates = list(full["candidate_boxes"])
            for roi_result in trial["rois"]:
                new_candidates.extend(roi_result["candidate_boxes"])
            new_value = {"box": outward_box(trial["selected_box"]), "candidate_boxes": new_candidates,
                         "selection_reason": trial["selection_reason"]}
            # Gold first enters here, after all predictions and ROI decisions.
            target = list(map(float, row["bbox"]))
            detail = {"id": row["id"], "text": row["text"], "image": sample["image"], "target_box": target,
                      "reference_count": len(plan.references), "plan": asdict(plan),
                      "current": scored(current_value, target), "semantic_full": scored(semantic_value, target),
                      "subject_full": scored(subject, target), "reference_first": scored(new_value, target),
                      "pipeline": trial, "latency_ms_all_arms": (time.perf_counter() - sample_started) * 1000}
            if index < args.overlays:
                preview = image.copy()
                for label, box, color in (("GT", target, (0, 180, 0)), ("CURRENT", current_value["box"], (255, 0, 0)),
                                           ("REF-FIRST", new_value["box"], (0, 0, 255))):
                    if box is not None:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(preview, label, (x1, max(18, y1)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                for reference in trial["references"]:
                    if reference["selected_box"] is not None:
                        x1, y1, x2, y2 = map(int, reference["selected_box"])
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (160, 160, 160), 1)
                for roi_result in trial["rois"]:
                    x1, y1, x2, y2 = map(int, roi_result["roi_xyxy"])
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 180, 255), 1)
                if not cv2.imwrite(str(args.output / f"public_{row['id']}.jpg"), preview):
                    raise OSError("Failed to save public overlay")
            details.append(detail)
            output.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            print(json.dumps({"progress": index + 1, "total": len(selected),
                              "current_iou": round(detail["current"]["iou"], 4),
                              "reference_first_iou": round(detail["reference_first"]["iou"], 4),
                              "selection_reason": trial["selection_reason"]}), flush=True)
    hashes_after = {path: sha256(Path(path)) for path in hashes_before}
    if hashes_before != hashes_after:
        raise RuntimeError("Source or model asset changed during the trial")
    gain = sum(detail["reference_first"]["iou"] >= .5 > detail["current"]["iou"] for detail in details)
    regression = sum(detail["current"]["iou"] >= .5 > detail["reference_first"]["iou"] for detail in details)
    return {"protocol": "public_reference_first_roi_trial_v1", "device": "cpu", "threads": args.threads,
            "coverage": coverage, "selected_reference_counts": [len(entry["plan"].references) for entry in selected],
            "gold_used_for_inference": False, "selection_sha256": sha256(selection_path),
            "source_and_asset_sha256": hashes_before, "source_and_asset_unchanged": True,
            "rrsis_train_val_image_id_overlap": 0,
            "detection_training_overlap": "unknown; original detector split/cache unavailable",
            "geographic_isolation": "unknown; this is a public development diagnostic, not independent test",
            "settings": {"detection_cap": 1600, "conf": .25, "nms_iou": .7, "max_det": 300,
                         "matching_temperature": .7, "current_spatial_weight": .15, "trial_spatial_weight": 0,
                         "crop_prompt": "complete original description", "subject_control": "parsed target phrase"},
            "model_load_ms": model_load_ms, "total_elapsed_ms": (time.perf_counter() - started) * 1000,
            "summary": {key: aggregate(details, key) for key in ("current", "semantic_full", "subject_full", "reference_first")},
            "new_correct_vs_current": gain, "lost_correct_vs_current": regression,
            "roi_inference_count": sum(len(detail["pipeline"]["rois"]) for detail in details),
            "selection_reasons": {reason: sum(detail["pipeline"]["selection_reason"] == reason for detail in details)
                                  for reason in sorted({detail["pipeline"]["selection_reason"] for detail in details})},
            "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overlays", type=int, default=4)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.threads <= 0 or args.overlays < 0:
        parser.error("limit and threads must be positive; overlays must be nonnegative")
    report = run(args)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": report.get("summary"),
                      "new_correct_vs_current": report.get("new_correct_vs_current"),
                      "lost_correct_vs_current": report.get("lost_correct_vs_current")}), flush=True)


if __name__ == "__main__":
    main()
