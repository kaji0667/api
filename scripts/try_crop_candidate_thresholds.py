"""Fixed public development diagnostic: fresh forwards at .25/.10/.05, then GT scoring.

This is a structure iteration after public dev feedback, not a holdout evaluation.
Prediction accepts only two SHA-bound public manifests and existing model assets.
All 195 memberships x three thresholds must finish before the separate score command.
No API dispatch, HTTP, downloads, search-ROI generation, training or model changes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "scripts"))
from try_crop_identity import (ANNOTATIONS_SHA256, _deadline, _iou, _outward,
                               _public_rows, check_deadline, load_json,
                               public_image, sha256, without_gold, write_new_json)

PROTOCOL = "public_fixed_candidate_threshold_crop_identity_v1"
THRESHOLDS = (.25, .10, .05)
ARMS = ("A", "E", "G")
DEADLINE_UTC = "2026-10-12T23:00:00Z"
COHORTS = {
    "public115": {"protocol": "public_native_full_candidates_frozen_v1", "count": 115,
                  "sha256": "522b9e091e764684d426f40a33be47575c3cd8b8ba3cc0b3d720de504a63f157"},
    "public80": {"protocol": "public_rrsisd_val_seed42_class4_prediction_selection_v1", "count": 80,
                 "sha256": "14de54bf759f4cb4aa36970748d9ff3f2105cfde3bfe24391ae5f3f0af4013d1"},
}
ASSETS = {
    "clip": ("openclip/ViT-L-14.pt", "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"),
    "detection": ("runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt", "f8de102ff9f8fdac01637520fb3cbb2e78ac490d0b318b3abdf60c5528a659df"),
}
FROZEN_PREVIOUS_SOURCES = {
    "tasks/detection/crop_identity.py": "d878c007933bdb1258ebd430a40e325e142502e490fe5ce47a667243e9e7c8e0",
    "scripts/try_crop_identity.py": "a90882883e81c8301dfadf3e3bd155dd9c1ef814bc0b9dc68a10267cc96a28c6",
    "tasks/detection/tests/test_crop_identity.py": "47560f1d21ff8267c24472e4262bd8a23aa5953310f86ae825e609e27cc499c8",
}


def load_inputs(manifest115: Path, manifest80: Path, data_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Keep two denominators; the five shared IDs are separate memberships."""
    members, manifests = [], {}
    for cohort, path in (("public115", manifest115), ("public80", manifest80)):
        spec = COHORTS[cohort]
        if sha256(path) != spec["sha256"]:
            raise ValueError("Fixed public manifest SHA differs")
        manifest = load_json(path)
        without_gold(manifest)
        samples = manifest.get("samples", [])
        if (manifest.get("protocol") != spec["protocol"] or len(samples) != spec["count"]
                or manifest.get("public_provenance", {}).get("dataset") != "public RRSIS-D val"
                or len({row["id"] for row in samples}) != len(samples)):
            raise ValueError("Invalid fixed public manifest")
        manifests[str(path.resolve())] = spec["sha256"]
        for row in samples:
            description = row.get("description") if cohort == "public115" else row.get("text")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Public description is missing")
            members.append({"cohort": cohort, "id": row["id"], "row_key": f"{cohort}:{row['id']}",
                            "image": str(public_image(row["image"], data_root)), "description": description,
                            "previous_025": ({key: row[key] for key in
                                               ("candidate_boxes", "baseline_index", "phrase_scores")}
                                              if cohort == "public115" else None)})
    if len(members) != 195 or len({row["id"] for row in members}) != 190:
        raise ValueError("Expected 195 memberships, including exactly five shared IDs")
    return members, manifests


def add_guarded_arm(prediction: dict[str, Any]) -> None:
    """G uses structural parser capability only; E remains unchanged."""
    use_e = prediction["parse_status"] == "parsed"
    source = "E" if use_e else "A"
    prediction["arms"]["G"] = dict(prediction["arms"][source])
    prediction["arms"]["G"].update(source_arm=source,
                                      guard_reason="parsed_target_identity" if use_e else "complete_description_fallback")
    prediction["G_used_E"] = use_e
    prediction["G_fallback_A"] = not use_e


def predict_one(detector: Any, scorer: Any, image: Any, member: dict[str, Any],
                threshold: float, image_hash: str) -> dict[str, Any]:
    import torch
    from tasks.detection.crop_identity import rank_full_candidates
    from tasks.detection.reference_plan import parse_reference_plan
    from native_detection import NativeDetection, sanitize_text
    from api_task_text import detection_text

    check_deadline(_deadline(DEADLINE_UTC))
    started = time.perf_counter()
    # Parsing precedes model inference, but this trial still runs the complete original sentence.
    plan = parse_reference_plan(member["description"])
    detector.candidate_confidence = threshold
    native_started = time.perf_counter()
    try:
        # Never reuse raw outputs: native NMS mutates its input xywh in place.
        native = detector.predict(image, member["description"], return_phrase_scores=True)
        boxes, baseline, phrase_scores = native.boxes.tolist(), native.grounding_index, native.phrase_scores.tolist()
        native_scores, diagnostics, native_status = native.scores.tolist(), native.diagnostics, "succeeded"
    except Exception as failure:
        boxes, baseline, phrase_scores, native_scores, diagnostics = [], None, [], [], None
        native_status = getattr(failure, "code", "native_prediction_failed")
    native_ms = (time.perf_counter() - native_started) * 1000
    check_deadline(_deadline(DEADLINE_UTC))
    try:
        baseline_scores = NativeDetection._position_adjusted_scores(
            SimpleNamespace(spatial_weight=.15),
            torch.tensor(phrase_scores, dtype=torch.float32).reshape(1, -1),
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), image.shape[:2],
            sanitize_text(detection_text(member["description"]), empty_placeholder=""),
        )[0].tolist()
        baseline_status = "original_native_absolute_primitive"
    except Exception:
        baseline_scores, baseline_status = None, "original_baseline_score_reconstruction_failed"
    crop_started = time.perf_counter()
    prediction = rank_full_candidates(scorer, image, member["description"], boxes, baseline,
                                      phrase_scores=phrase_scores, hybrid=True, baseline_scores=baseline_scores)
    crop_ms = (time.perf_counter() - crop_started) * 1000
    if (prediction["parse_status"] == "parsed") != (plan is not None):
        raise RuntimeError("Parser changed during a prediction")
    add_guarded_arm(prediction)
    prediction.update(cohort=member["cohort"], id=member["id"], row_key=member["row_key"],
                      candidate_confidence=threshold, image=member["image"], image_sha256=image_hash,
                      image_hw=list(image.shape[:2]), native_status=native_status,
                      native_confidence_scores=native_scores, native_diagnostics=diagnostics,
                      baseline_score_status=baseline_status, native_latency_ms=native_ms, crop_latency_ms=crop_ms,
                      latency_ms=(time.perf_counter() - started) * 1000,
                      completed_utc=datetime.now(timezone.utc).isoformat())
    old = member.get("previous_025")
    if threshold == .25 and old is not None:
        prediction["original_025_audit"] = {
            "candidate_order_exact": boxes == old["candidate_boxes"],
            "baseline_index_exact": baseline == old["baseline_index"],
            "phrase_scores_exact": phrase_scores == old["phrase_scores"],
        }
        failed = [key for key, equal in prediction["original_025_audit"].items() if not equal]
        if failed:
            raise RuntimeError(f"Original .25 public115 baseline drift at {member['row_key']}: {','.join(failed)}")
    without_gold(prediction)
    return prediction


def validate_prediction_rows(rows: list[dict[str, Any]], frozen: dict[str, Any], completion: dict[str, Any]) -> None:
    if (frozen.get("protocol") != PROTOCOL or completion.get("protocol") != PROTOCOL
            or not completion.get("completed") or frozen.get("thresholds") != list(THRESHOLDS)
            or frozen.get("arms") != list(ARMS) or frozen.get("membership_count") != 195
            or frozen.get("unique_id_count") != 190 or completion.get("prediction_count") != 585
            or completion.get("membership_count") != 195 or completion.get("unique_id_count") != 190
            or frozen.get("deadline_utc") != DEADLINE_UTC
            or _deadline(completion["completed_utc"]) >= _deadline(DEADLINE_UTC)):
        raise ValueError("Only all 585 fixed predictions completed before the deadline can be scored")
    if (len(frozen.get("manifest_sha256", {})) != 2
            or set(frozen["manifest_sha256"].values()) != {spec["sha256"] for spec in COHORTS.values()}):
        raise ValueError("Both original fixed public manifests must be registered")
    registered, completed = _deadline(frozen["registered_utc"]), _deadline(completion["completed_utc"])
    if registered is None or completed is None or registered > completed:
        raise ValueError("Invalid run registration/completion timeline")
    memberships = frozen["memberships"]
    expected_members = {(row["cohort"], row["id"]) for row in memberships}
    if (len(memberships) != 195 or len(expected_members) != 195
            or len({row["id"] for row in memberships}) != 190
            or any(sum(row["cohort"] == cohort for row in memberships) != spec["count"]
                   for cohort, spec in COHORTS.items())):
        raise ValueError("Incomplete public cohort membership registration")
    keys = [(row["cohort"], row["id"], row["candidate_confidence"]) for row in rows]
    expected = {(cohort, ident, threshold) for cohort, ident in expected_members for threshold in THRESHOLDS}
    if len(rows) != 585 or len(set(keys)) != 585 or set(keys) != expected:
        raise ValueError("All two-cohort/three-threshold memberships are required before GT access")
    for row in rows:
        without_gold(row)
        row_completed = _deadline(row["completed_utc"])
        if (row["row_key"] != f"{row['cohort']}:{row['id']}" or row_completed is None
                or not registered <= row_completed <= completed):
            raise ValueError("Invalid prediction registration or completion time")
        boxes = row["candidate_boxes"]
        if (len(boxes) > 300 or any(len(box) != 4 or not all(math.isfinite(float(v)) for v in box) for box in boxes)
                or len(row["phrase_scores"]) != len(boxes) or len(row["native_confidence_scores"]) != len(boxes)):
            raise ValueError("Invalid native candidate pool")
        diagnostics = row["native_diagnostics"]
        if row["native_status"] == "succeeded" and (not isinstance(diagnostics, dict)
                or diagnostics["confidence_threshold"] != row["candidate_confidence"]
                or diagnostics["nms_candidate_count"] != len(boxes)):
            raise ValueError("Native diagnostics must match the actual threshold and pool")
        if row["parse_status"] not in {"parsed", "complete_description_fallback"}:
            raise ValueError("Unknown parser capability state")
        use_e = row["parse_status"] == "parsed"
        source = "E" if use_e else "A"
        if (row["G_used_E"] is not use_e or row["G_fallback_A"] is not (not use_e)
                or row["arms"]["G"]["source_arm"] != source):
            raise ValueError("G must depend only on parser capability")
        for arm in ARMS:
            record = row["arms"][arm]
            index = record["selected_index"]
            if index is not None and (type(index) is not int or not 0 <= index < len(boxes)):
                raise ValueError("Arm selection is outside its real candidate pool")
            if record["selected_box_xyxy"] != (boxes[index] if index is not None else None):
                raise ValueError("Arm selected coordinates must retain the native candidate")
        if any(row["arms"]["G"][key] != row["arms"][source][key]
               for key in ("selected_index", "selected_box_xyxy", "status")):
            raise ValueError("G changed its source arm selection")


def predict(args: argparse.Namespace) -> dict[str, Any]:
    deadline = _deadline(DEADLINE_UTC)
    check_deadline(deadline)
    members, manifests = load_inputs(args.manifest115, args.manifest80, args.data_root)
    assets = {key: args.models_dir.expanduser().resolve() / value[0] for key, value in ASSETS.items()}
    for key, path in assets.items():
        if not path.is_file() or sha256(path) != ASSETS[key][1]:
            raise ValueError("Required existing model asset differs from its registered SHA")
    for relative, expected in FROZEN_PREVIOUS_SOURCES.items():
        if sha256(CODE_DIR / relative) != expected:
            raise ValueError("Previous crop implementation must remain frozen")
    sources = [Path(__file__), *(CODE_DIR / relative for relative in FROZEN_PREVIOUS_SOURCES),
               CODE_DIR / "tasks/detection/reference_plan.py", CODE_DIR / "participant_api_starter-main/native_detection.py",
               CODE_DIR / "participant_api_starter-main/grounding_spatial.py", CODE_DIR / "participant_api_starter-main/api_task_text.py",
               CODE_DIR / "ultralytics/nn/modules/text_backbone.py", CODE_DIR / "text_encoder/matching.py",
               CODE_DIR / "ultralytics/utils/ops.py"]
    frozen_hashes = {str(path): sha256(path) for path in [*sources, *assets.values()]}
    frozen_hashes.update(manifests)
    image_hashes = {row["image"]: sha256(Path(row["image"])) for row in members}
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("predictions.jsonl", "frozen_run.json", "completion.json")):
        raise ValueError("Preserve existing output; choose a fresh directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HFSA_API_DIAGNOSTICS"] = "1"
    import cv2
    import torch
    from tasks.detection.crop_identity import FrozenCLIPCropScorer
    sys.path.insert(0, str(CODE_DIR / "participant_api_starter-main"))
    from native_detection import NativeDetection, NativeTextEncoder

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    check_deadline(deadline)
    registered = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    encoder = NativeTextEncoder(device=args.device, pretrained=str(assets["clip"]))
    scorer = FrozenCLIPCropScorer(encoder.encoder.model, encoder.encode_descriptions, batch_size=args.crop_batch_size)
    detector = NativeDetection(assets["detection"], device=args.device, encoder=encoder,
                               max_image_size=1600, spatial_weight=.15, candidate_confidence=.25)
    model_load_ms = (time.perf_counter() - started) * 1000
    frozen = {
        "protocol": PROTOCOL, "development_status": "public_dev_feedback_structure_iteration",
        "registered_utc": registered, "deadline_utc": DEADLINE_UTC, "annotations_sha256": ANNOTATIONS_SHA256,
        "source_and_asset_sha256": frozen_hashes, "manifest_sha256": manifests,
        "image_sha256": image_hashes, "membership_count": 195, "unique_id_count": 190,
        "memberships": [{key: row[key] for key in ("cohort", "id", "row_key")} for row in members],
        "thresholds": list(THRESHOLDS), "arms": list(ARMS), "prediction_count": 585,
        "device": args.device, "threads": args.threads, "torch_seed": 0, "torch_version": torch.__version__,
        "package_versions": {name: version(name) for name in ("torchvision", "open_clip_torch", "Pillow")},
        "model_load_ms": model_load_ms, "gold_used_for_inference": False,
        "configuration": {"crop_batch_size": args.crop_batch_size, "crop_cache": False,
                          "fresh_complete_forward_per_threshold": True, "model_instances": 1, "clip_instances": 1,
                          "candidate_confidence": list(THRESHOLDS), "nms_iou": .7, "max_det": 300, "max_nms": 30000,
                          "detection_cap": 1600, "native_spatial_weight": .15, "matching_temperature": .7,
                          "hybrid_weight": .15, "contrast_normalization": "independent_pool_minmax",
                          "hybrid_formula": "raw_native_phrase_plus_original_native_absolute_0.15_plus_0.15_contrast_pool_minmax",
                          "hybrid_failure_policy": "explicit_original_selection_fallback",
                          "G_rule": "parsed_uses_E_else_complete_description_fallback_uses_A",
                          "preprocess": scorer.preprocess_configuration,
                          "maximum_candidate_crops": 175500, "extra_candidate_crops_upper_bound": 117000,
                          "maximum_detector_forwards": 585},
    }
    without_gold(frozen)
    check_deadline(deadline)
    write_new_json(args.output / "frozen_run.json", frozen)
    rows = []
    try:
        with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as output:
            for member in members:
                check_deadline(deadline)
                image = cv2.imread(member["image"])
                if image is None:
                    raise ValueError("Public image decode failed; run remains incomplete")
                for threshold in THRESHOLDS:
                    prediction = predict_one(detector, scorer, image, member, threshold, image_hashes[member["image"]])
                    check_deadline(deadline)
                    output.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    rows.append(prediction)
                    print(json.dumps({"progress": len(rows), "total": 585, "row_key": member["row_key"],
                                      "threshold": threshold, "candidates": len(prediction["candidate_boxes"]),
                                      "pre_nms": (prediction["native_diagnostics"] or {}).get("pre_nms_above_threshold"),
                                      "score_status": prediction["score_status"]}), flush=True)
    finally:
        detector.candidate_confidence = .25
    check_deadline(deadline)
    if frozen_hashes != {path: sha256(Path(path)) for path in frozen_hashes}:
        raise RuntimeError("Frozen source, asset or manifest changed during inference")
    if image_hashes != {path: sha256(Path(path)) for path in image_hashes}:
        raise RuntimeError("Public image changed during inference")
    check_deadline(deadline)
    completion = {"completed": True, "protocol": PROTOCOL, "membership_count": 195, "unique_id_count": 190,
                  "prediction_count": len(rows), "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "candidate_count": sum(len(row["candidate_boxes"]) for row in rows),
                  "reported_encoded_crop_count": sum(row.get("encoded_crop_count", 0) for row in rows),
                  "predictions_sha256": sha256(args.output / "predictions.jsonl"),
                  "frozen_run_sha256": sha256(args.output / "frozen_run.json"),
                  "elapsed_ms": (time.perf_counter() - started) * 1000, "gold_used_for_inference": False}
    validate_prediction_rows(rows, frozen, completion)
    check_deadline(deadline)
    write_new_json(args.output / "completion.json", completion)
    return completion


def _distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"count": len(values), "total": sum(values), "min": ordered[0], "median": statistics.median(values),
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1], "max": ordered[-1], "mean": statistics.mean(values)}


def _transitions(rows: list[dict[str, Any]], arm: str, comparator: dict[str, dict[str, Any]],
                 comparator_arm: str, level: float) -> dict[str, Any]:
    gains = [row["id"] for row in rows if row["arms"][arm]["iou"] >= level > comparator[row["id"]]["arms"][comparator_arm]["iou"]]
    losses = [row["id"] for row in rows if comparator[row["id"]]["arms"][comparator_arm]["iou"] >= level > row["arms"][arm]["iou"]]
    return {"gain_count": len(gains), "loss_count": len(losses), "gain_ids": gains, "loss_ids": losses}


def score_run(run_dir: Path, annotations: Path) -> dict[str, Any]:
    score_started_utc = datetime.now(timezone.utc).isoformat()
    completion, frozen = load_json(run_dir / "completion.json"), load_json(run_dir / "frozen_run.json")
    if (completion["predictions_sha256"] != sha256(run_dir / "predictions.jsonl")
            or completion["frozen_run_sha256"] != sha256(run_dir / "frozen_run.json")):
        raise ValueError("Complete predictions or frozen registration changed")
    predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    validate_prediction_rows(predictions, frozen, completion)
    # All completeness/registration checks above precede the first GT file access.
    if frozen["annotations_sha256"] != ANNOTATIONS_SHA256 or sha256(annotations) != ANNOTATIONS_SHA256:
        raise ValueError("Expected the fixed public validation annotations")
    gold = {row["id"]: row["bbox"] for row in _public_rows(annotations)}
    details = []
    for row in predictions:
        target = gold[row["id"]]
        if (len(target) != 4 or not all(math.isfinite(float(value)) for value in target)
                or target[2] <= target[0] or target[3] <= target[1]):
            raise ValueError("Invalid public xyxy target")
        oracle = max((_iou(_outward(box, row["image_hw"]), target) for box in row["candidate_boxes"]), default=0.)
        diag = row["native_diagnostics"] or {}
        details.append({"cohort": row["cohort"], "id": row["id"], "row_key": row["row_key"],
                        "candidate_confidence": row["candidate_confidence"], "parse_status": row["parse_status"],
                        "candidate_count": len(row["candidate_boxes"]), "oracle_iou": oracle,
                        "pre_nms_count": diag.get("pre_nms_above_threshold"), "raw_anchor_count": diag.get("raw_anchor_count"),
                        "native_status": row["native_status"], "score_status": row["score_status"],
                        "latency_ms": row["latency_ms"], "native_latency_ms": row["native_latency_ms"],
                        "crop_latency_ms": row["crop_latency_ms"], "encoded_crop_count": row.get("encoded_crop_count", 0),
                        "image_encode_batch_count": row.get("image_encode_batch_count", 0),
                        "G_used_E": row["G_used_E"], "G_fallback_A": row["G_fallback_A"],
                        "arms": {arm: {"iou": _iou(_outward(row["arms"][arm]["selected_box_xyxy"], row["image_hw"]), target),
                                       "valid": row["arms"][arm]["selected_index"] is not None,
                                       "selected_index": row["arms"][arm]["selected_index"]} for arm in ARMS}})
    summaries = {}
    for cohort in COHORTS:
        base_rows = {row["id"]: row for row in details if row["cohort"] == cohort and row["candidate_confidence"] == .25}
        summaries[cohort] = {}
        for threshold in THRESHOLDS:
            rows = [row for row in details if row["cohort"] == cohort and row["candidate_confidence"] == threshold]
            n = len(rows)
            same_rows = {row["id"]: row for row in rows}
            arms = {}
            for arm in ARMS:
                arms[arm] = {"sample_count": n, "valid_count": sum(row["arms"][arm]["valid"] for row in rows),
                             "mean_iou": statistics.mean(row["arms"][arm]["iou"] for row in rows),
                             "candidate_misses_at_0_5": sum(row["oracle_iou"] < .5 for row in rows),
                             "selection_misses_at_0_5": sum(row["oracle_iou"] >= .5 > row["arms"][arm]["iou"] for row in rows)}
                for level in (.5, .7):
                    arms[arm][f"correct_at_{level}"] = sum(row["arms"][arm]["iou"] >= level for row in rows)
                    arms[arm][f"versus_same_threshold_A_at_{level}"] = _transitions(rows, arm, same_rows, "A", level)
                    for base_arm in ("A", "E"):
                        arms[arm][f"versus_0.25_{base_arm}_at_{level}"] = _transitions(rows, arm, base_rows, base_arm, level)
            pre_counts = [row["pre_nms_count"] for row in rows if row["pre_nms_count"] is not None]
            summaries[cohort][str(threshold)] = {
                "sample_count": n, "arms": arms,
                "oracle_at_0_5": sum(row["oracle_iou"] >= .5 for row in rows),
                "oracle_at_0_7": sum(row["oracle_iou"] >= .7 for row in rows),
                "mean_oracle_iou": statistics.mean(row["oracle_iou"] for row in rows),
                "candidate_count": _distribution([row["candidate_count"] for row in rows]),
                "pre_nms_count": _distribution(pre_counts) if pre_counts else None,
                "at_300_candidate_cap": sum(row["candidate_count"] == 300 for row in rows),
                "above_300_pre_nms": sum(value > 300 for value in pre_counts),
                "above_30000_pre_nms": sum(value > 30000 for value in pre_counts),
                "native_failure_count": sum(row["native_status"] != "succeeded" for row in rows),
                "crop_failure_count": sum(row["score_status"] not in {"succeeded", "empty_candidates"} for row in rows),
                "G_used_E": sum(row["G_used_E"] for row in rows), "G_fallback_A": sum(row["G_fallback_A"] for row in rows),
                "latency_ms": _distribution([row["latency_ms"] for row in rows]),
                "native_ms_total": sum(row["native_latency_ms"] for row in rows),
                "crop_ms_total": sum(row["crop_latency_ms"] for row in rows),
                "reported_encoded_crops": sum(row["encoded_crop_count"] for row in rows),
                "reported_image_encode_batches": sum(row["image_encode_batch_count"] for row in rows),
            }
    return {"protocol": PROTOCOL, "development_status": "public_dev_feedback_structure_iteration",
            "score_started_utc": score_started_utc, "summary": summaries, "details": details,
            "completion": completion, "prediction_sha256": completion["predictions_sha256"],
            "gold_used_for_inference": False,
            "training_overlap": "Detection RSVG-ALL original split/cache unknown; public development diagnostic",
            "geographic_isolation": "unknown; public square samples do not validate large image/ABC/lawn accuracy"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("predict", help="Fresh forward for every cohort/sample/threshold; no GT argument")
    inference.add_argument("--manifest115", type=Path, required=True)
    inference.add_argument("--manifest80", type=Path, required=True)
    inference.add_argument("--models-dir", type=Path, required=True)
    inference.add_argument("--data-root", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    inference.add_argument("--threads", type=int, default=4)
    inference.add_argument("--crop-batch-size", type=int, default=8)
    evaluation = commands.add_parser("score", help="GT access after all fixed 585 predictions completed unchanged")
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--annotations", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        if args.threads <= 0 or args.crop_batch_size <= 0:
            parser.error("threads and crop-batch-size must be positive")
        print(json.dumps(predict(args)), flush=True)
    else:
        result = score_run(args.run_dir, args.annotations)
        write_new_json(args.output, result)
        print(json.dumps({"metrics": str(args.output), "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
