"""Fixed 195 public memberships: whole-object crop and complete-text fallback.

Two preregistered arms, with exactly the original .25 candidate pool and .15
native absolute/crop weights. No API calls, official inputs, new boxes or GT in
predict. Only score reads SHA-bound public annotations after completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "participant_api_starter-main"))
from scripts.try_crop_candidate_thresholds import load_inputs, ASSETS
from scripts.try_crop_identity import ANNOTATIONS_SHA256, sha256, write_new_json


def predict(args):
    from tasks.detection.whole_crop_trial import WholeBBoxCLIPScorer, rank_whole_crop
    members, manifests = load_inputs(args.manifest115, args.manifest80, args.data_root)
    baseline_receipt = json.loads((args.baseline.parent / "completion.json").read_text())
    if (baseline_receipt.get("completed") is not True
            or sha256(args.baseline) != baseline_receipt["predictions_sha256"]):
        raise ValueError("Baseline must be complete and unchanged")
    old = {row["row_key"]: row for row in map(json.loads, args.baseline.read_text().splitlines())
           if row["candidate_confidence"] == .25}
    if set(old) != {row["row_key"] for row in members}:
        raise ValueError("Baseline denominator differs")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), CODE / "tasks/detection/whole_crop_trial.py",
               CODE / "tasks/detection/reference_plan.py", CODE / "tasks/detection/crop_identity.py",
               CODE / "scripts/try_crop_candidate_thresholds.py", CODE / "scripts/try_crop_identity.py",
               CODE / "participant_api_starter-main/native_detection.py", args.baseline,
               CODE / "openclip/ViT-L-14.pt"]
    hashes = {str(path): sha256(path) for path in sources}
    if hashes[str(CODE / "openclip/ViT-L-14.pt")] != ASSETS["clip"][1]:
        raise ValueError("Original CLIP differs")
    images = {row["image"]: sha256(Path(row["image"])) for row in members}
    write_new_json(args.output / "frozen.json", {
        "protocol": "same_native_pool_whole_crop_v1", "count": 195, "unique_ids": 190,
        "registered_utc": datetime.now(timezone.utc).isoformat(), "manifests": manifests,
        "sources_assets_baseline": hashes, "images": images, "device": args.device,
        "arms": {"pad_guarded": "square-pad original role cues; unsupported keeps original G",
                 "pad_full": "square-pad original role cues; unsupported scores complete description"},
        "weights": {"native_absolute": .15, "crop_minmax": .15},
        "padding": "square, centered, rounded CLIP mean RGB, original transform",
        "prediction_reads_gold": False, "production_enabled": False,
    })
    snapshots = args.output / "source_snapshot"
    snapshots.mkdir()
    for path in sources:
        if path.suffix == ".py":
            shutil.copy2(path, snapshots / path.name)
    import cv2
    import torch
    from native_detection import NativeTextEncoder
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    encoder = NativeTextEncoder(args.device, pretrained=str(CODE / "openclip/ViT-L-14.pt"))
    scorer = WholeBBoxCLIPScorer(encoder.encoder.model, encoder.encode_descriptions)
    with (args.output / "predictions.jsonl").open("x") as output:
        for index, member in enumerate(members):
            prior = old[member["row_key"]]
            result = rank_whole_crop(scorer, cv2.imread(member["image"]), member["description"],
                prior["candidate_boxes"], prior["baseline_adjusted_scores"],
                prior["arms"]["A"]["selected_index"], prior["arms"]["G"]["selected_index"])
            arms = {arm: prior["arms"][arm]["selected_box_xyxy"] for arm in ("A", "G")}
            for arm in ("pad_guarded", "pad_full"):
                chosen = result[arm + "_index"]
                arms[arm] = prior["candidate_boxes"][chosen] if chosen is not None else None
            output.write(json.dumps({"cohort": member["cohort"], "id": member["id"],
                                     "arms": arms, "matching": result}, allow_nan=False) + "\n")
            output.flush()
            if (index + 1) % 25 == 0:
                print(json.dumps({"progress": index + 1, "total": 195}), flush=True)
    if any(sha256(Path(path)) != digest for path, digest in {**hashes, **images}.items()):
        raise ValueError("Registered file changed")
    write_new_json(args.output / "completion.json", {"completed": True, "count": 195,
        "prediction_sha256": sha256(args.output / "predictions.jsonl"),
        "frozen_sha256": sha256(args.output / "frozen.json"), "seconds": time.perf_counter() - started})


def score(args):
    if sha256(args.annotations) != ANNOTATIONS_SHA256:
        raise ValueError("Public annotations differ")
    from scripts.try_grounding_text import score as score_complete_run
    score_complete_run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("predict")
    for name in ("manifest115", "manifest80", "data-root", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cpu")
    s = commands.add_parser("score")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    (predict if args.command == "predict" else score)(args)


if __name__ == "__main__":
    main()
