"""Fixed public 115+80 comparison; predictions finish before gold-only scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "participant_api_starter-main"))
from scripts.try_crop_candidate_thresholds import load_inputs, ASSETS
from scripts.try_crop_identity import sha256, write_new_json, ANNOTATIONS_SHA256
from scripts.try_reference_grounding import box_iou, outward_box


def predict(args):
    import torch
    from native_detection import NativeDetection, NativeTextEncoder
    from api_task_text import detection_text
    from tasks.detection.spatial_reference_grounding import (
        SpatialReferenceGroundingExecutor, parse_spatial_reference_plan,
    )
    members, manifests = load_inputs(args.manifest115, args.manifest80, args.data_root)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), CODE / "tasks/detection/spatial_reference_grounding.py",
             CODE / "tasks/detection/reference_grounding.py", CODE / "tasks/detection/reference_plan.py",
             CODE / "tasks/detection/text_conditioning.py", CODE / "text_encoder/matching.py",
             CODE / "participant_api_starter-main/native_detection.py",
             CODE / "participant_api_starter-main/api_task_text.py",
             CODE / "participant_api_starter-main/grounding_spatial.py"]
    assets = {name: CODE / value[0] for name, value in ASSETS.items()}
    for name, path in assets.items():
        if sha256(path) != ASSETS[name][1]:
            raise ValueError("Original model asset SHA changed")
    protected = {str(p): sha256(p) for p in [*paths, *assets.values()]}
    for path in paths:
        target = args.output / "source_snapshot" / path.relative_to(CODE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    plans = [parse_spatial_reference_plan(row["description"]) for row in members]
    registration = {
        "count": 195, "unique_ids": 190, "manifests": manifests, "sources_assets": protected,
        "annotations_sha256": ANNOTATIONS_SHA256,
        "positioned_reference_count": sum(p is not None and p.absolute_position is not None for p in plans),
        "multiple_reference_count": sum(p is not None and len(p.references) > 1 for p in plans),
        "parsed_count": sum(p is not None for p in plans), "device": args.device,
        "spatial_weight": .15, "candidate_confidence": .25, "seed": 0,
        "samples": [{"row_key": row["row_key"], "image_sha256": sha256(Path(row["image"])),
                     "plan": asdict(plan) if plan else None} for row, plan in zip(members, plans)],
    }
    write_new_json(args.output / "registration.json", registration)
    print(json.dumps({key: registration[key] for key in
                      ("count", "parsed_count", "positioned_reference_count", "multiple_reference_count")}), flush=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    encoder = NativeTextEncoder(device=args.device, pretrained=str(assets["clip"]))
    detector = NativeDetection(assets["detection"], device=args.device, encoder=encoder,
                              max_image_size=1600, spatial_weight=.15, candidate_confidence=.25)
    call_records = []

    class RecordedDetector:
        def predict(self, image, question, **kwargs):
            value = detector.predict(image, question, **kwargs)
            call_records.append({"query": question, "hw": list(value.original_size),
                                 "candidates": value.boxes.tolist(), "kwargs": kwargs})
            return value

    executor = SpatialReferenceGroundingExecutor(RecordedDetector(), description_extractor=detection_text,
                                                 image_converter=detector._image)
    def value(result):
        index = result.grounding_index
        return {"box": outward_box(result.boxes[index]) if index is not None else None,
                "index": index, "candidate_boxes": result.boxes.tolist()}

    started = time.monotonic()
    with (args.output / "predictions.jsonl").open("x", encoding="utf8") as stream:
        for i, row in enumerate(members):
            image = detector._image(row["image"])
            call_records.clear()
            trial = executor.predict(image, row["description"])
            trace = list(call_records)
            baseline = detector.predict(image, row["description"])
            stream.write(json.dumps({"row_key": row["row_key"], "cohort": row["cohort"], "id": row["id"],
                                     "native": value(baseline), "spatial_reference": value(trial),
                                     "diagnostics": trial.diagnostics, "calls": trace}, allow_nan=False) + "\n")
            stream.flush()
            if (i + 1) % 20 == 0:
                print(f"Public predictions {i+1}/195", flush=True)
    if any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError("Source or asset changed during prediction")
    write_new_json(args.output / "completion.json", {
        "completed": True, "count": 195, "seconds": time.monotonic() - started,
        "predictions_sha256": sha256(args.output / "predictions.jsonl"),
        "registration_sha256": sha256(args.output / "registration.json"),
    })


def score(args):
    from scripts.try_crop_identity import _public_rows
    completed = json.loads((args.output / "completion.json").read_text(encoding="utf8"))
    if (not completed["completed"] or completed["count"] != 195
            or completed["predictions_sha256"] != sha256(args.output / "predictions.jsonl")
            or completed["registration_sha256"] != sha256(args.output / "registration.json")
            or sha256(args.annotations) != ANNOTATIONS_SHA256):
        raise ValueError("Complete unchanged predictions and fixed public annotations required")
    rows = [json.loads(line) for line in (args.output / "predictions.jsonl").read_text(encoding="utf8").splitlines()]
    if len(rows) != 195 or len({row["row_key"] for row in rows}) != 195:
        raise ValueError("Denominator changed")
    gold = {row["id"]: row["bbox"] for row in _public_rows(args.annotations)}
    report = {"completion": completed, "cohorts": {}}
    for cohort, count in (("public115", 115), ("public80", 80)):
        selected = [row for row in rows if row["cohort"] == cohort]
        if len(selected) != count:
            raise ValueError("Cohort changed")
        arms = {arm: [box_iou(row[arm]["box"], gold[row["id"]]) for row in selected]
                for arm in ("native", "spatial_reference")}
        report["cohorts"][cohort] = {
            "count": count,
            **{arm: {"correct_at_0_5": sum(iou >= .5 for iou in values),
                     "correct_at_0_7": sum(iou >= .7 for iou in values),
                     "mean_iou": sum(values) / count} for arm, values in arms.items()},
            "gain": sum(a < .5 <= b for a, b in zip(arms["native"], arms["spatial_reference"])),
            "loss": sum(b < .5 <= a for a, b in zip(arms["native"], arms["spatial_reference"])),
            "outcomes": dict(Counter(row["diagnostics"]["grounding_outcome"] for row in selected)),
            "reasons": dict(Counter(row["diagnostics"]["grounding_reason"] for row in selected)),
        }
    write_new_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("predict")
    for name in ("manifest115", "manifest80", "data-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cpu")
    s = commands.add_parser("score")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    predict(args) if args.command == "predict" else score(args)
