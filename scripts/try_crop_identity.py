"""Separate public selection, fixed-candidate inference and post-hoc GT scoring.

No HTTP, official requests, training, downloads, ROI candidate generation or
production dispatch. prepare115 projects cached predictions through a whitelist;
prepare80 reproduces the original public seed42/class4 selection without gold.
predict never accepts annotation metadata. score only accepts completed runs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

PROTOCOL = "public_frozen_full_original_crop_identity_v1"
SELECTION115_SHA256 = "c5672ed0aec3488afa221be0167d5097bc9cc40d0f424af4d81cc238bfe08f46"
ANNOTATIONS_SHA256 = "6059b3d058ee75daf3e49027280d140af0fe5a1b6e2d0f983a14bb96cdb90a61"
AUDIT_MANIFESTS = {
    "public_native_full_candidates_frozen_v1": (
        "522b9e091e764684d426f40a33be47575c3cd8b8ba3cc0b3d720de504a63f157", "cached_public115_full_original", 115),
    "public_rrsisd_val_seed42_class4_prediction_selection_v1": (
        "14de54bf759f4cb4aa36970748d9ff3f2105cfde3bfe24391ae5f3f0af4013d1", "public80_seed42_class4", 80),
}
AUDIT_CANDIDATE_ASSET_SHA256 = "fc8be91dc66b1295d2995cefa152c4f7f17bd8f8ac6b4430589ad0b42a724f9d"
FORBIDDEN = {"bbox", "segmentation", "target_box", "iou", "oracle_iou", "class_idx", "class_name",
             "annotation_bbox_xyxy", "gold", "ground_truth"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def without_gold(value: Any) -> None:
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("Gold/metric/category fields cannot enter a prediction manifest")
        for child in value.values():
            without_gold(child)
    elif isinstance(value, list):
        for child in value:
            without_gold(child)


def prepare115(selection_path: Path, report_path: Path) -> dict[str, Any]:
    if sha256(selection_path) != SELECTION115_SHA256:
        raise ValueError("Expected the frozen, original public 115 selection")
    selection, report = load_json(selection_path), load_json(report_path)
    if (report.get("protocol") != "public_target_query_candidate_trial_v1"
            or report.get("selection_sha256") != SELECTION115_SHA256
            or len(selection["samples"]) != 115 or len(report["details"]) != 115):
        raise ValueError("Cached report must match the fixed public 115 selection")
    details = {row["id"]: row for row in report["details"]}
    if len(details) != 115:
        raise ValueError("Duplicate cached public IDs")
    samples = []
    for entry in selection["samples"]:
        row = details[entry["id"]]
        original = row["pipeline"]["full_original"]
        if (row["text"] != entry["plan"]["original_description"] or row["image"] != entry["image"]
                or row["current"]["candidate_boxes"] != original["candidate_boxes"]):
            raise ValueError("Original description/image/candidate identity differs")
        anchors = [{"description": anchor["description"], "relation": anchor["relation"],
                    "selected_box": anchor["selected_box"], "source": "cached_native_reference_prediction"}
                   for anchor in row["pipeline"]["references"]]
        samples.append({"id": entry["id"], "image": entry["image"], "description": row["text"],
                        "candidate_boxes": original["candidate_boxes"], "baseline_index": row["current"]["selected_index"],
                        "phrase_scores": original["phrase_scores"], "anchors": anchors})
    payload = {"protocol": PROTOCOL, "source": "cached_public115_full_original", "sample_count": 115,
               "annotations_sha256": selection["annotations_sha256"],
               "source_selection_sha256": sha256(selection_path), "source_report_sha256": sha256(report_path),
               "selection_rule": "unchanged original115; inference candidates only, no GT selection",
               "candidate_asset_sha256": {key: value for key, value in report["source_and_asset_sha256"].items()
                                           if key.endswith(".pt")}, "samples": samples}
    without_gold(payload)
    return payload


def _public_rows(annotations: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("split") != "val" for row in rows) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected unique public RRSIS-D validation rows")
    return rows


def public_image(raw_path: str, data_root: Path) -> Path:
    root = data_root.expanduser().resolve()
    path = Path(raw_path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Image must be an existing file under the public data root")
    return path


def prepare80(annotations: Path, data_root: Path) -> dict[str, Any]:
    rows = _public_rows(annotations)
    categories = sorted({int(row["class_idx"]) for row in rows})
    if len(categories) != 20:
        raise ValueError("Expected the original20 public RRSIS-D categories")
    rng = random.Random(42)
    selected = []
    for category in categories:
        pool = sorted((row for row in rows if int(row["class_idx"]) == category), key=lambda row: str(row["id"]))
        if len(pool) < 4:
            raise ValueError("Not enough validation rows for the fixed class4 selection")
        selected.extend(sorted(rng.sample(pool, 4), key=lambda row: str(row["id"])))
    samples = [{"id": row["id"], "image": str(public_image(row["image"], data_root)), "description": row["text"]}
               for row in selected]
    payload = {"protocol": PROTOCOL, "source": "public80_seed42_class4", "sample_count": 80,
               "annotations_sha256": sha256(annotations), "selection_rule": "Random42; class ascending; sorted IDs; four per class",
               "samples": samples}
    without_gold(payload)
    return payload


def _deadline(value: str | None) -> datetime | None:
    if value is None:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Deadline requires an explicit timezone")
    return result.astimezone(timezone.utc)


def check_deadline(deadline: datetime | None) -> None:
    if deadline is not None and datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("Public prediction deadline reached; run remains incomplete")


def predict(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    without_gold(manifest)
    if manifest.get("protocol") in AUDIT_MANIFESTS:
        expected, source, count = AUDIT_MANIFESTS[manifest["protocol"]]
        if (sha256(args.manifest) != expected or len(manifest["samples"]) != count
                or manifest["public_provenance"].get("dataset") != "public RRSIS-D val"):
            raise ValueError("Independent audit manifest must match its frozen public SHA")
        samples = []
        for entry in manifest["samples"]:
            item = dict(entry)
            if source == "public80_seed42_class4":
                item["description"] = item.pop("text")
            samples.append(item)
        manifest = {"protocol": PROTOCOL, "source": source, "sample_count": count,
                    "annotations_sha256": ANNOTATIONS_SHA256, "samples": samples}
        if source == "cached_public115_full_original":
            if args.candidate_provenance is None:
                raise ValueError("Cached audit candidates require their no-gold model provenance JSON")
            if sha256(args.candidate_provenance) != AUDIT_CANDIDATE_ASSET_SHA256:
                raise ValueError("Independent cached model provenance must match its frozen SHA")
            manifest["candidate_asset_sha256"] = load_json(args.candidate_provenance)
            without_gold(manifest["candidate_asset_sha256"])
    if (manifest.get("protocol") != PROTOCOL or manifest.get("source") not in
            {"cached_public115_full_original", "public80_seed42_class4"}
            or len(manifest["samples"]) != manifest["sample_count"]
            or len({entry["id"] for entry in manifest["samples"]}) != len(manifest["samples"])):
        raise ValueError("Invalid frozen public manifest")
    deadline = _deadline(args.deadline_utc)
    check_deadline(deadline)
    samples = manifest["samples"][:args.smoke_first] if args.smoke_first else manifest["samples"]
    paths = [public_image(entry["image"], args.data_root) for entry in samples]
    models_root = args.models_dir.expanduser().resolve()
    assets = {"clip": models_root / "openclip/ViT-L-14.pt",
              "detection": models_root / "runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt"}
    generate = manifest["source"] == "public80_seed42_class4"
    for asset in assets.values():
        if not asset.is_file():
            raise FileNotFoundError("Required existing model asset is missing")
    candidate_hashes = manifest.get("candidate_asset_sha256", {})
    if not generate and (not isinstance(candidate_hashes, dict) or len(candidate_hashes) != 2):
        raise ValueError("Cached candidate provenance requires original detector and CLIP hashes")
    normalized_assets = {}
    for old_path, expected in candidate_hashes.items():
        # Cached115 predictions require the same local CLIP and detector assets.
        old_path = old_path.replace("\\", "/")
        if old_path.endswith("/openclip/ViT-L-14.pt"):
            key = "clip"
        elif old_path.endswith("/runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt"):
            key = "detection"
        else:
            raise ValueError("Unexpected cached model asset")
        if key in normalized_assets:
            raise ValueError("Duplicate cached model asset")
        normalized_assets[key] = expected
        asset = assets[key]
        if sha256(asset) != expected:
            raise ValueError("Cached candidate model provenance differs from current assets")
    if not generate and set(normalized_assets) != {"clip", "detection"}:
        raise ValueError("Cached candidate provenance must cover both assets")
    sources = [Path(__file__), CODE_DIR / "tasks/detection/crop_identity.py", CODE_DIR / "tasks/detection/reference_plan.py",
               CODE_DIR / "participant_api_starter-main/native_detection.py", CODE_DIR / "ultralytics/nn/modules/text_backbone.py",
               CODE_DIR / "participant_api_starter-main/grounding_spatial.py", CODE_DIR / "participant_api_starter-main/api_task_text.py",
               CODE_DIR / "text_encoder/matching.py"]
    provenance_sources = [args.candidate_provenance] if args.candidate_provenance is not None else []
    frozen_hashes = {str(path): sha256(path) for path in [*sources, *assets.values(), args.manifest, *provenance_sources]}
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("predictions.jsonl", "frozen_run.json", "completion.json")):
        raise ValueError("Preserve the existing trial; choose another output directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import cv2
    import torch
    from tasks.detection.crop_identity import FrozenCLIPCropScorer, rank_full_candidates
    from tasks.detection.reference_plan import parse_reference_plan

    sys.path.insert(0, str(CODE_DIR / "participant_api_starter-main"))
    from native_detection import NativeDetection, NativeTextEncoder, sanitize_text
    from api_task_text import detection_text

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    check_deadline(deadline)
    started = time.perf_counter()
    encoder = NativeTextEncoder(device=args.device, pretrained=str(assets["clip"]))
    scorer = FrozenCLIPCropScorer(encoder.encoder.model, encoder.encode_descriptions, batch_size=args.crop_batch_size)
    detector = (NativeDetection(assets["detection"], device=args.device, encoder=encoder,
                               max_image_size=1600, spatial_weight=.15) if generate else None)
    model_load_ms = (time.perf_counter() - started) * 1000
    frozen = {"protocol": PROTOCOL, "manifest_sha256": sha256(args.manifest),
              "registered_utc": datetime.now(timezone.utc).isoformat(),
              "annotations_sha256": manifest["annotations_sha256"], "source": manifest["source"],
              "sample_count": len(samples), "requested_sample_count": manifest["sample_count"],
              "smoke_first": args.smoke_first, "source_and_asset_sha256": frozen_hashes,
              "device": args.device, "threads": args.threads, "torch_seed": 0,
              "torch_version": torch.__version__, "deadline_utc": args.deadline_utc,
              "arms": ["A", "B", "C", *(["D"] if args.geometry else []), *(["E"] if args.hybrid else [])],
              "configuration": {"crop_batch_size": args.crop_batch_size, "geometry_weight": .15 if args.geometry else 0,
                                "conf": .25, "nms_iou": .7, "max_det": 300, "detection_cap": 1600,
                                "native_spatial_weight": .15, "matching_temperature": .7,
                                "hybrid_weight": .15 if args.hybrid else 0.,
                                "hybrid_formula": "raw_native_phrase_plus_original_native_absolute_0.15_plus_0.15_contrast_pool_minmax",
                                "hybrid_failure_policy": "explicit_original_selection_fallback",
                                "preprocess": scorer.preprocess_configuration}, "model_load_ms": model_load_ms,
              "gold_used_for_inference": False, "no_added_candidates": True}
    write_new_json(args.output / "frozen_run.json", frozen)
    candidate_count = 0
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as output:
        for index, (entry, path) in enumerate(zip(samples, paths)):
            check_deadline(deadline)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Public image decode failed")
            current_started = time.perf_counter()
            description = entry["description"]
            # Identity/relationship parse precedes every model call.
            plan = parse_reference_plan(description)
            native_status = "cached_prediction"
            anchors = entry.get("anchors", [])
            if detector is not None:
                try:
                    native = detector.predict(image, description, return_phrase_scores=True)
                    boxes, baseline, phrase_scores = native.boxes.tolist(), native.grounding_index, native.phrase_scores.tolist()
                    native_status = "succeeded"
                except Exception as failure:
                    boxes, baseline, phrase_scores = [], None, []
                    native_status = getattr(failure, "code", "native_prediction_failed")
                if args.geometry and plan is not None:
                    anchors = []
                    for ref in plan.references:
                        check_deadline(deadline)
                        try:
                            anchor = detector.predict(image, ref.description)
                            selected_box = (anchor.boxes[anchor.grounding_index].tolist()
                                            if anchor.grounding_index is not None else None)
                            status = "succeeded" if selected_box is not None else "empty_reference"
                        except Exception:
                            selected_box, status = None, "reference_prediction_failed"
                        anchors.append({"description": ref.description, "relation": ref.relation,
                                        "selected_box": selected_box, "status": status})
            else:
                boxes, baseline, phrase_scores = entry["candidate_boxes"], entry["baseline_index"], entry["phrase_scores"]
            baseline_scores = None
            baseline_score_status = "not_requested"
            if args.hybrid and phrase_scores is not None:
                try:
                    score_tensor = torch.tensor(phrase_scores, dtype=torch.float32).reshape(1, -1)
                    box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                    adjusted = NativeDetection._position_adjusted_scores(
                        SimpleNamespace(spatial_weight=.15), score_tensor, box_tensor, image.shape[:2],
                        sanitize_text(detection_text(description), empty_placeholder=""),
                    )
                    baseline_scores = adjusted[0].tolist()
                    baseline_score_status = "original_native_absolute_primitive"
                except Exception:
                    baseline_score_status = "original_baseline_score_reconstruction_failed"
            prediction = rank_full_candidates(scorer, image, description, boxes, baseline,
                                              phrase_scores=phrase_scores, geometry=args.geometry, anchors=anchors,
                                              hybrid=args.hybrid, baseline_scores=baseline_scores)
            prediction.update(id=entry["id"], image=str(path), image_sha256=sha256(path), image_hw=list(image.shape[:2]),
                              native_status=native_status, latency_ms=(time.perf_counter() - current_started) * 1000,
                              anchors=anchors if args.geometry else [], baseline_score_status=baseline_score_status)
            without_gold(prediction)
            output.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            candidate_count += len(boxes)
            print(json.dumps({"progress": index + 1, "total": len(samples), "candidates": len(boxes),
                              "score_status": prediction["score_status"]}), flush=True)
    check_deadline(deadline)
    if frozen_hashes != {path: sha256(Path(path)) for path in frozen_hashes}:
        raise RuntimeError("A frozen source/model/manifest changed during inference")
    completion = {"completed": True, "protocol": PROTOCOL, "sample_count": len(samples),
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "candidate_count": candidate_count, "predictions_sha256": sha256(args.output / "predictions.jsonl"),
                  "frozen_run_sha256": sha256(args.output / "frozen_run.json"),
                  "elapsed_ms": (time.perf_counter() - started) * 1000, "gold_used_for_inference": False}
    write_new_json(args.output / "completion.json", completion)
    return completion


def _outward(box: Any, hw: list[int]) -> list[int] | None:
    if box is None:
        return None
    h, w = hw
    a, b, c, d = [min(max(float(value), 0), extent) for value, extent in zip(box, (w, h, w, h))]
    return [math.floor(a), math.floor(b), math.ceil(c), math.ceil(d)] if c > a and d > b else None


def _iou(box: Any, target: list[float]) -> float:
    if box is None:
        return 0.
    overlap = max(0., min(box[2], target[2]) - max(box[0], target[0])) * max(0., min(box[3], target[3]) - max(box[1], target[1]))
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (target[2] - target[0]) * (target[3] - target[1]) - overlap)
    return overlap / union if union > 0 else 0.


def score_run(run_dir: Path, annotations: Path) -> dict[str, Any]:
    score_started_utc = datetime.now(timezone.utc).isoformat()
    completion, frozen = load_json(run_dir / "completion.json"), load_json(run_dir / "frozen_run.json")
    if (not completion.get("completed") or completion.get("protocol") != PROTOCOL
            or completion["predictions_sha256"] != sha256(run_dir / "predictions.jsonl")
            or completion["frozen_run_sha256"] != sha256(run_dir / "frozen_run.json")
            or frozen["annotations_sha256"] != sha256(annotations)):
        raise ValueError("Scoring requires a complete, unchanged run and matching public annotations")
    predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(predictions) != completion["sample_count"] or len({row["id"] for row in predictions}) != len(predictions):
        raise ValueError("Incomplete or duplicate prediction rows")
    for row in predictions:
        without_gold(row)
    # This independent post-hoc command is the only scoring path that reads GT.
    gold = {row["id"]: row["bbox"] for row in _public_rows(annotations)}
    details = []
    for row in predictions:
        target = gold[row["id"]]
        if (len(target) != 4 or not all(math.isfinite(float(value)) for value in target)
                or target[2] <= target[0] or target[3] <= target[1]):
            raise ValueError("Invalid public xyxy target box")
        oracle = max((_iou(_outward(box, row["image_hw"]), target) for box in row["candidate_boxes"]), default=0.)
        details.append({"id": row["id"], "parse_status": row["parse_status"], "candidate_count": len(row["candidate_boxes"]),
                        "oracle_iou": oracle, "arms": {key: {"iou": _iou(_outward(value["selected_box_xyxy"], row["image_hw"]), target),
                                                           "valid": value["selected_index"] is not None}
                                                     for key, value in row["arms"].items()}})
    n = len(details)
    summary = {}
    for arm in frozen["arms"]:
        values = [row["arms"][arm] for row in details]
        summary[arm] = {"sample_count": n, "valid_count": sum(value["valid"] for value in values),
                        "correct_at_0_5": sum(value["iou"] >= .5 for value in values),
                        "correct_at_0_7": sum(value["iou"] >= .7 for value in values),
                        "mean_iou": sum(value["iou"] for value in values) / n,
                        "candidate_misses_at_0_5": sum(row["oracle_iou"] < .5 for row in details),
                        "selection_misses_at_0_5": sum(row["oracle_iou"] >= .5 and row["arms"][arm]["iou"] < .5 for row in details)}
        for threshold in (.5, .7):
            gains = [row["id"] for row in details if row["arms"][arm]["iou"] >= threshold > row["arms"]["A"]["iou"]]
            losses = [row["id"] for row in details if row["arms"]["A"]["iou"] >= threshold > row["arms"][arm]["iou"]]
            summary[arm][f"transition_at_{threshold}"] = {"gain_count": len(gains), "loss_count": len(losses),
                                                         "gain_ids": gains, "loss_ids": losses}
    return {"protocol": PROTOCOL, "prediction_sha256": completion["predictions_sha256"], "summary": summary,
            "score_started_utc": score_started_utc,
            "candidate_recall_at_0_5": sum(row["oracle_iou"] >= .5 for row in details) / n,
            "candidate_recall_at_0_7": sum(row["oracle_iou"] >= .7 for row in details) / n,
            "parsed_count": sum(row["parse_status"] == "parsed" for row in details),
            "gold_used_for_inference": False, "details": details, "completion": completion,
            "training_overlap": "Detection RSVG-ALL original split/cache unknown; public development diagnostic",
            "geographic_isolation": "unknown; these public800-square samples do not validate large image/ABC/lawn accuracy"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cached = commands.add_parser("prepare115", help="Whitelist frozen old candidates; no model inference")
    cached.add_argument("--source-selection", type=Path, required=True)
    cached.add_argument("--source-report", type=Path, required=True)
    cached.add_argument("--output", type=Path, required=True)
    balanced = commands.add_parser("prepare80", help="Freeze originalseed42/class4 IDs; no prediction/GT selection")
    balanced.add_argument("--annotations", type=Path, required=True)
    balanced.add_argument("--data-root", type=Path, required=True)
    balanced.add_argument("--output", type=Path, required=True)
    inference = commands.add_parser("predict", help="No annotation argument; completed predictions contain no gold")
    inference.add_argument("--manifest", type=Path, required=True)
    inference.add_argument("--candidate-provenance", type=Path, help="No-gold source-model SHA mapping for cached audit115")
    inference.add_argument("--models-dir", type=Path, required=True)
    inference.add_argument("--data-root", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    inference.add_argument("--threads", type=int, default=4)
    inference.add_argument("--crop-batch-size", type=int, default=8)
    inference.add_argument("--geometry", action="store_true", help="SeparateD arm, fixed C+.15 soft geometry; no hardROI")
    inference.add_argument("--hybrid", action="store_true", help="SeparateE arm: originalraw phrase+nativeposition+.15 contrast minmax")
    inference.add_argument("--smoke-first", type=int, default=0, help="Only firstN frozen samples; marked as partial smoke")
    inference.add_argument("--deadline-utc", help="Optional UTC/offset ISO deadline; refuses to mark an interrupted run complete")
    evaluation = commands.add_parser("score", help="Read public GT only after prediction completion")
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--annotations", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare115":
        result = prepare115(args.source_selection, args.source_report)
        write_new_json(args.output, result)
        print(json.dumps({"manifest": str(args.output), "sample_count": result["sample_count"]}))
    elif args.command == "prepare80":
        result = prepare80(args.annotations, args.data_root)
        write_new_json(args.output, result)
        print(json.dumps({"manifest": str(args.output), "sample_count": result["sample_count"]}))
    elif args.command == "predict":
        if args.threads <= 0 or args.crop_batch_size <= 0 or args.smoke_first < 0:
            parser.error("threads/batch_size must be positive; smoke_first nonnegative")
        print(json.dumps(predict(args)), flush=True)
    else:
        result = score_run(args.run_dir, args.annotations)
        write_new_json(args.output, result)
        print(json.dumps({"metrics": str(args.output), "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
