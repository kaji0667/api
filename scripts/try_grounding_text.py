"""Frozen public text-condition ablation; gold is read by score only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "participant_api_starter-main"))
from scripts.try_crop_candidate_thresholds import load_inputs, ASSETS
from scripts.try_crop_identity import sha256, write_new_json, _outward, _iou


def predict(args):
    import cv2
    import numpy as np
    import torch
    from native_detection import NativeDetection, NativeTextEncoder
    from tasks.detection.crop_identity import FrozenCLIPCropScorer, rank_full_candidates
    from tasks.detection.text_conditioning import RoleConditionedEncoder, parse_grounding_text
    members, manifests = load_inputs(args.manifest115, args.manifest80, args.data_root)
    old = {row["row_key"]: row for row in map(json.loads, args.baseline.read_text().splitlines())
           if row["candidate_confidence"] == .25}
    if set(old) != {row["row_key"] for row in members}:
        raise ValueError("Baseline denominator differs")
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), CODE / "tasks/detection/text_conditioning.py",
             CODE / "tasks/detection/reference_plan.py", CODE / "tasks/detection/crop_identity.py",
             CODE / "participant_api_starter-main/native_detection.py", args.baseline]
    for relative, expected in ASSETS.values():
        path = CODE / relative
        if sha256(path) != expected:
            raise ValueError("Original asset differs")
        files.append(path)
    hashes = {str(path): sha256(path) for path in files}
    images = {row["image"]: sha256(Path(row["image"])) for row in members}
    write_new_json(args.output / "frozen.json", {
        "protocol": "source_bound_grounding_text_trial_v1", "weights": [.25, .5],
        "registered_utc": datetime.now(timezone.utc).isoformat(), "count": len(members),
        "sources_assets_baseline": hashes, "manifests": manifests, "images": images,
        "device": args.device, "threads": 4, "prediction_reads_gold": False,
    })
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.perf_counter()
    encoder = NativeTextEncoder(args.device, pretrained=str(CODE / "openclip/ViT-L-14.pt"))
    native = NativeDetection(CODE / "runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt",
                            args.device, encoder=encoder, max_image_size=1600, spatial_weight=.15)
    scorer = FrozenCLIPCropScorer(encoder.encoder.model, encoder.encode_descriptions)
    print(json.dumps({"load_seconds": time.perf_counter()-started}), flush=True)
    with (args.output / "predictions.jsonl").open("x") as target:
        for index, row in enumerate(members):
            image = cv2.imread(row["image"])
            source = old[row["row_key"]]
            original_boxes = np.asarray(source["candidate_boxes"], dtype=np.float32).reshape(-1, 4)
            plan = parse_grounding_text(row["description"])
            record = {"cohort":row["cohort"], "id":row["id"], "plan":bool(plan), "arms":{
                "A":source["arms"]["A"]["selected_box_xyxy"],
                "G":source["arms"]["G"]["selected_box_xyxy"]}}
            trial_started = time.perf_counter()
            for weight in (.25, .5):
                label = str(weight)
                if plan is None:
                    for arm in ("native", "guarded", "original_pool"):
                        record["arms"][label+":"+arm] = record["arms"]["G"]
                    continue
                native.encoder = RoleConditionedEncoder(encoder, target_weight=weight)
                result = native.predict(image, row["description"], return_phrase_scores=True,
                                        score_boxes_xyxy=original_boxes)
                selected = result.grounding_index
                record["arms"][label+":native"] = result.boxes[selected].tolist() if selected is not None else None
                ranking = rank_full_candidates(scorer, image, row["description"], result.boxes, selected,
                    phrase_scores=result.phrase_scores, hybrid=True, baseline_scores=result.baseline_adjusted_scores)
                record["arms"][label+":guarded"] = ranking["arms"]["E"]["selected_box_xyxy"]
                record[label+":candidates"] = result.boxes.tolist()
                if len(original_boxes):
                    # Re-score the original pool in the same new full-image maps.
                    position = np.asarray(source["baseline_adjusted_scores"]) - np.asarray(source["phrase_scores"])
                    correction = np.array([r.get("contrast_pool_minmax", 0.) for r in source["crop_scores"]])
                    if len(correction) != len(original_boxes):
                        correction = np.zeros(len(original_boxes))
                    scores = result.extra_phrase_scores + position + .15 * correction
                    winner = int(np.argmax(scores))
                    record["arms"][label+":original_pool"] = original_boxes[winner].tolist()
                else:
                    record["arms"][label+":original_pool"] = None
            record["latency_ms"] = (time.perf_counter()-trial_started)*1000
            target.write(json.dumps(record, allow_nan=False)+"\n")
            target.flush()
            print(json.dumps({"progress":index+1, "total":len(members), "parsed":bool(plan),
                              "ms":round(record["latency_ms"])}), flush=True)
    if any(sha256(Path(path)) != expected for path,expected in {**hashes, **images}.items()):
        raise ValueError("Registered file changed")
    write_new_json(args.output / "completion.json", {"completed":True, "count":len(members),
        "prediction_sha256":sha256(args.output/"predictions.jsonl"),
        "frozen_sha256":sha256(args.output/"frozen.json"), "seconds":time.perf_counter()-started})


def score(args):
    receipt = json.loads((args.output / "completion.json").read_text())
    if (receipt.get("completed") is not True or sha256(args.output/"predictions.jsonl") != receipt["prediction_sha256"]
            or sha256(args.output/"frozen.json") != receipt["frozen_sha256"]):
        raise ValueError("Incomplete or modified prediction")
    gold = {r["id"]: r["bbox"] for r in map(json.loads,args.annotations.read_text().splitlines())}
    rows = list(map(json.loads,(args.output/"predictions.jsonl").read_text().splitlines()))
    if len(rows) != receipt["count"] or len({(r["cohort"],r["id"]) for r in rows}) != len(rows):
        raise ValueError("Prediction denominator differs")
    details, metrics = [], {}
    for row in rows:
        details.append({"cohort":row["cohort"], "id":row["id"], "ious":{
            arm:_iou([math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])],
                     gold[row["id"]]) if box is not None else 0.
            for arm,box in row["arms"].items()}})
    for cohort in ("public115", "public80"):
        group = [row for row in details if row["cohort"] == cohort]
        metrics[cohort] = {}
        for arm in group[0]["ious"]:
            metrics[cohort][arm] = {"n":len(group),
                "at50":sum(row["ious"][arm]>=.5 for row in group),
                "at70":sum(row["ious"][arm]>=.7 for row in group),
                "mean_iou":sum(row["ious"][arm] for row in group)/len(group),
                "gain":sum(row["ious"][arm]>=.5>row["ious"]["G"] for row in group),
                "loss":sum(row["ious"]["G"]>=.5>row["ious"][arm] for row in group)}
    write_new_json(args.output/"metrics.json", {"metrics":metrics, "details":details,
        "score_source_sha256":sha256(Path(__file__)),
        "limits":"iterated public development; unknown training/geographic overlap; no official score"})
    print(json.dumps(metrics,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("predict")
    for name in ("manifest115", "manifest80", "data-root", "baseline", "output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cpu")
    s=commands.add_parser("score")
    s.add_argument("--output",type=Path,required=True)
    s.add_argument("--annotations",type=Path,required=True)
    args=parser.parse_args()
    (predict if args.command=="predict" else score)(args)


if __name__=="__main__":
    main()
