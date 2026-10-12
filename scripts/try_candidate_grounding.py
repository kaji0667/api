"""Fixed public trial of target queries, full/ROI candidates and soft relations.

Gold is accessed for scoring only after all prediction arms. This CLI reuses
the original detector and never starts HTTP or reads official requests.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from scripts.try_reference_grounding import (
    aggregate, box_iou, image_path, outward_box, prediction_value,
    public_selection, scored, sha256,
)


def control_value(record: dict) -> dict:
    return {
        "box": outward_box(record["selected_box"]),
        "candidate_boxes": record["candidate_boxes"],
        "selected_index": record["selected_index"],
    }


def arm_summary(rows: list[dict]) -> dict:
    keys = ("current", "semantic_full", "subject_full", "union_semantic", "reference_candidate")
    result = {key: aggregate(rows, key) for key in keys}
    for key in keys:
        result[key]["correct_at_0_7"] = sum(row[key]["iou"] >= .7 for row in rows)
    return result


def transitions(rows: list[dict], proposed: str, baseline: str, threshold: float) -> dict:
    gains = [row["id"] for row in rows if row[proposed]["iou"] >= threshold > row[baseline]["iou"]]
    losses = [row["id"] for row in rows if row[baseline]["iou"] >= threshold > row[proposed]["iou"]]
    return {"gain_count": len(gains), "loss_count": len(losses), "gain_ids": gains, "loss_ids": losses}


def run(args: argparse.Namespace) -> dict:
    selected, coverage = public_selection(args.annotations, args.limit)
    train_rows = [json.loads(line) for line in args.train_annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not train_rows or any(row.get("split") != "train" for row in train_rows):
        raise ValueError("Expected public training metadata")
    if {entry["row"]["image_id"] for entry in selected} & {row["image_id"] for row in train_rows}:
        raise ValueError("Public selected validation images overlap training image IDs")
    manifest = [{"id": entry["row"]["id"], "image": str(image_path(entry["row"], args.data_root)),
                 "plan": asdict(entry["plan"])} for entry in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.selection_only and (args.output / "results.jsonl").exists():
        raise ValueError("Preserve the existing trial; choose another output directory")
    selection_path = args.output / "selection.json"
    selection_content = json.dumps({
        "selection": "supported syntax, ascending IDs, fixed first N; no gold or prediction selection",
        "annotations_sha256": sha256(args.annotations), "coverage": coverage,
        "samples": manifest,
    }, ensure_ascii=False, indent=2) + "\n"
    if selection_path.exists():
        if selection_path.read_text(encoding="utf-8") != selection_content:
            raise ValueError("Frozen selection changed; preserve this trial")
    else:
        selection_path.write_text(selection_content, encoding="utf-8")
    print(json.dumps({"selected_count": len(selected), **coverage}), flush=True)
    if args.selection_only:
        return {"selection_only": True, "coverage": coverage, "samples": manifest}

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, str(CODE_DIR / "participant_api_starter-main"))
    from native_detection import NativeDetection, NativeTextEncoder
    from tasks.detection.candidate_grounding import predict_candidate_grounding

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    models_root = args.models_dir.expanduser().resolve()
    assets = {"detection": models_root / "runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt",
              "clip": models_root / "openclip/ViT-L-14.pt"}
    sources = [Path(__file__), CODE_DIR / "scripts/try_reference_grounding.py",
               CODE_DIR / "tasks/detection/reference_plan.py", CODE_DIR / "tasks/detection/reference_grounding.py",
               CODE_DIR / "tasks/detection/candidate_grounding.py",
               CODE_DIR / "participant_api_starter-main/native_detection.py",
               CODE_DIR / "participant_api_starter-main/grounding_spatial.py",
               CODE_DIR / "text_encoder/matching.py", CODE_DIR / "text_encoder/phrase_classifier.py",
               CODE_DIR / "text_encoder/model.py", CODE_DIR / "ultralytics/nn/modules/text_backbone.py"]
    hashes_before = {str(path): sha256(path) for path in [*sources, *assets.values()]}
    started = time.perf_counter()
    print("Loading unchanged local detection/CLIP assets on CPU", flush=True)
    encoder = NativeTextEncoder(device="cpu", pretrained=str(assets["clip"]))
    detector = NativeDetection(assets["detection"], device="cpu", encoder=encoder,
                               max_image_size=1600, spatial_weight=0)
    model_load_ms = (time.perf_counter() - started) * 1000
    details = []
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as output:
        for index, (entry, sample) in enumerate(zip(selected, manifest)):
            row, plan = entry["row"], entry["plan"]
            image = cv2.imread(sample["image"])
            if image is None:
                raise ValueError("Public image decode failed")
            sample_started = time.perf_counter()
            trial = predict_candidate_grounding(detector, image, plan)
            semantic = control_value(trial["full_original"])
            subject = control_value(trial["full_subject"])
            detector.spatial_weight = .15
            try:
                current = prediction_value(detector.predict(image, plan.original_description))
            finally:
                detector.spatial_weight = 0
            if not np.array_equal(np.asarray(current["candidate_boxes"]), np.asarray(semantic["candidate_boxes"])):
                raise AssertionError("Current/full-original candidate generation differs")
            all_boxes = [candidate["box"] for candidate in trial["pool"]]
            active_boxes = [candidate["box"] for candidate in trial["pool"] if not candidate["suppressed"]]
            union = {"box": outward_box(trial["unconstrained_selected_box"]), "candidate_boxes": active_boxes}
            proposed = {"box": outward_box(trial["selected_box"]), "candidate_boxes": active_boxes,
                        "selection_reason": trial["selection_reason"]}
            # Gold enters only after all actual prediction arms and decisions.
            target = list(map(float, row["bbox"]))
            detail = {
                "id": row["id"], "text": row["text"], "image": sample["image"],
                "target_box": target, "reference_count": len(plan.references), "plan": asdict(plan),
                "current": scored(current, target), "semantic_full": scored(semantic, target),
                "subject_full": scored(subject, target), "union_semantic": scored(union, target),
                "reference_candidate": scored(proposed, target), "pipeline": trial,
                "all_pool_oracle_iou": max((box_iou(outward_box(box), target) for box in all_boxes), default=0),
                "all_pool_candidate_count": len(all_boxes), "active_pool_candidate_count": len(active_boxes),
                "latency_ms_all_arms": (time.perf_counter() - sample_started) * 1000,
            }
            if index < args.overlays:
                preview = image.copy()
                for label, box, color in (("GT", target, (0, 180, 0)), ("CURRENT", current["box"], (255, 0, 0)),
                                           ("CANDIDATE", proposed["box"], (0, 0, 255))):
                    if box is not None:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(preview, label, (x1, max(18, y1)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                for reference in trial["references"]:
                    if reference["selected_box"] is not None:
                        x1, y1, x2, y2 = map(int, reference["selected_box"])
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (160, 160, 160), 1)
                if trial["roi_record"] is not None:
                    x1, y1, x2, y2 = map(int, trial["roi_record"]["roi_xyxy"])
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 180, 255), 1)
                if not cv2.imwrite(str(args.output / f"public_{row['id']}.jpg"), preview):
                    raise OSError("Failed to save overlay")
            details.append(detail)
            output.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            print(json.dumps({"progress": index + 1, "total": len(selected),
                              "current_iou": round(detail["current"]["iou"], 4),
                              "subject_iou": round(detail["subject_full"]["iou"], 4),
                              "proposed_iou": round(detail["reference_candidate"]["iou"], 4)}), flush=True)
    hashes_after = {path: sha256(Path(path)) for path in hashes_before}
    if hashes_before != hashes_after:
        raise RuntimeError("Source or asset changed during inference")
    first = details[:min(12, len(details))]
    additional = details[12:]
    return {
        "protocol": "public_target_query_candidate_trial_v1", "device": "cpu", "threads": args.threads,
        "torch_seed": 0, "coverage": coverage, "sample_count": len(details),
        "selected_reference_counts": [len(entry["plan"].references) for entry in selected],
        "gold_used_for_inference": False, "source_and_asset_unchanged": True,
        "source_and_asset_sha256": hashes_before, "selection_sha256": sha256(selection_path),
        "rrsis_train_val_selected_image_id_overlap": 0,
        "detection_training_overlap": "unknown; original RSVG-ALL split/cache missing",
        "geographic_isolation": "unknown; public development diagnostic",
        "settings": {"detection_cap": 1600, "conf": .25, "nms_iou": .7, "max_det": 300,
                     "matching_temperature": .7, "current_spatial_weight": .15, "trial_spatial_weight": 0,
                     "target_query": "parsed target phrase with original attributes",
                     "all_pool_scoring": "same full_subject forward phrase maps"},
        "model_load_ms": model_load_ms, "total_elapsed_ms": (time.perf_counter() - started) * 1000,
        "summary": arm_summary(details), "previous_first_12_summary": arm_summary(first),
        "additional_samples_summary": arm_summary(additional) if additional else None,
        "transitions": {str(threshold): {baseline: transitions(details, "reference_candidate", baseline, threshold)
                                           for baseline in ("current", "subject_full", "union_semantic")}
                        for threshold in (.5, .7)},
        "all_pool_recall_at_0_5": sum(row["all_pool_oracle_iou"] >= .5 for row in details) / len(details),
        "all_pool_candidate_count": sum(row["all_pool_candidate_count"] for row in details),
        "active_pool_candidate_count": sum(row["active_pool_candidate_count"] for row in details),
        "roi_inference_count": sum(row["pipeline"]["roi_record"] is not None for row in details),
        "selection_reasons": {reason: sum(row["pipeline"]["selection_reason"] == reason for row in details)
                              for reason in sorted({row["pipeline"]["selection_reason"] for row in details})},
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=115)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overlays", type=int, default=4)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.threads <= 0 or args.overlays < 0:
        parser.error("limit/threads must be positive and overlays nonnegative")
    report = run(args)
    path = args.output / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "summary": report.get("summary")}), flush=True)


if __name__ == "__main__":
    main()
