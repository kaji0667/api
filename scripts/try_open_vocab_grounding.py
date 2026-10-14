"""Predict on frozen public manifests; score only a completed, hash-bound run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from tasks.detection.open_vocab_grounding import OpenVocabularyGrounder, box_iou, file_sha256

PUBLIC_MANIFESTS = {
    "522b9e091e764684d426f40a33be47575c3cd8b8ba3cc0b3d720de504a63f157": "RRSIS115",
    "14de54bf759f4cb4aa36970748d9ff3f2105cfde3bfe24391ae5f3f0af4013d1": "RRSIS80",
    "57c1d7180e8ca47b217342c1f62d38b507cdeba19143716142fe2714ec481cac": "RefSegRS5",
}


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def predict(args: argparse.Namespace) -> None:
    from PIL import Image
    import torch
    samples, files = [], [Path(__file__), CODE_DIR/"tasks/detection/open_vocab_grounding.py",
                          CODE_DIR/"tasks/detection/reference_plan.py"]
    for path in args.manifest:
        cohort = PUBLIC_MANIFESTS.get(file_sha256(path))
        if cohort is None:
            raise ValueError("Only previously frozen public inputs are authorized for this runner")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload["samples"]:
            samples.append({"cohort": cohort, "id": entry["id"], "image": entry["image"],
                            "description": entry.get("description", entry.get("text"))})
        files.append(path)
    if len({(row["cohort"],row["id"]) for row in samples}) != len(samples):
        raise ValueError("Duplicate cohort membership")
    args.output.mkdir(parents=True, exist_ok=False)
    images = {row["image"]: file_sha256(Path(row["image"])) for row in samples}
    files += list(args.model_dir.glob("*"))
    hashes = {str(path): file_sha256(path) for path in files if path.is_file()}
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    configuration = {"threshold": .25, "spatial_weight": .15, "tiled": True,
                     "device": args.device, "threads": args.threads, "seed": 0,
                     "query": "separate target/reference token spans; complete text if unsupported"}
    write_json(args.output/"frozen.json", {"protocol": "public_open_vocab_trial_v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "source_model_input_sha256": hashes, "image_sha256": images, "samples": samples,
        "torch_version": torch.__version__, "prediction_reads_gold": False})
    print(json.dumps({"registered": len(samples), "device": args.device}), flush=True)
    started = time.perf_counter()
    model = OpenVocabularyGrounder(args.model_dir, device=args.device)
    print(json.dumps({"load_seconds": time.perf_counter()-started}), flush=True)
    with (args.output/"predictions.jsonl").open("x", encoding="utf-8") as target:
        for index,row in enumerate(samples):
            sample_started = time.perf_counter()
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
            result = model.predict(image, row["description"])
            record = {**row, "prediction": result, "latency_ms": (time.perf_counter()-sample_started)*1000}
            target.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            target.flush()
            print(json.dumps({"progress": index+1, "total": len(samples), "cohort": row["cohort"],
                              "candidates": len(result["candidates"]), "ms": round(record["latency_ms"])}), flush=True)
    if any(file_sha256(Path(path)) != expected for path,expected in {**hashes,**images}.items()):
        raise ValueError("Registered source/model/image changed during inference")
    write_json(args.output/"completion.json", {"completed": True, "count": len(samples),
        "prediction_sha256": file_sha256(args.output/"predictions.jsonl"),
        "frozen_sha256": file_sha256(args.output/"frozen.json"), "elapsed_seconds": time.perf_counter()-started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0})


def outward(box: list[float] | None) -> list[int] | None:
    import math
    return [math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3])] if box else None


def score(args: argparse.Namespace) -> None:
    complete = json.loads((args.output/"completion.json").read_text())
    if (complete.get("completed") is not True
            or file_sha256(args.output/"predictions.jsonl") != complete["prediction_sha256"]
            or file_sha256(args.output/"frozen.json") != complete["frozen_sha256"]):
        raise ValueError("Incomplete or modified trial cannot be scored")
    gold = {row["id"]: row["bbox"] for row in map(json.loads, args.annotations.read_text(encoding="utf-8").splitlines())}
    rows = [json.loads(line) for line in (args.output/"predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != complete["count"]:
        raise ValueError("Prediction denominator mismatch")
    details = []
    for row in rows:
        if row["cohort"] == "RefSegRS5":
            import numpy as np
            from PIL import Image
            with Image.open(args.mask_dir/f"{row['id']}.tif") as source:
                mask = np.asarray(source)
            if mask.ndim == 3:
                if not np.all(mask == mask[..., :1]):
                    raise ValueError("Mask channels disagree")
                mask = mask[...,0]
            if not set(np.unique(mask)).issubset({0,1,255}):
                raise ValueError("Unknown public mask encoding")
            ys,xs = np.nonzero(mask)
            if not len(xs):
                raise ValueError("Empty public gold mask")
            target = [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        else:
            target = gold[row["id"]]
        prediction = row["prediction"]
        final, raw = outward(prediction["box"]), outward(prediction["raw_box"])
        oracle = max((box_iou(outward(item["box"]),target) for item in prediction["candidates"]), default=0.)
        details.append({"cohort": row["cohort"], "id": row["id"], "iou": box_iou(final,target) if final else 0.,
                        "raw_iou": box_iou(raw,target) if raw else 0., "oracle_iou": oracle,
                        "latency_ms": row["latency_ms"]})
    metrics = {}
    for cohort in sorted({row["cohort"] for row in rows}):
        group = [item for item in details if item["cohort"] == cohort]
        metrics[cohort] = {"count":len(group), "correct_at_0_5":sum(r["iou"]>=.5 for r in group),
            "correct_at_0_7":sum(r["iou"]>=.7 for r in group),"raw_correct_at_0_5":sum(r["raw_iou"]>=.5 for r in group),
            "oracle_at_0_5":sum(r["oracle_iou"]>=.5 for r in group),"mean_iou":sum(r["iou"] for r in group)/len(group),
            "mean_latency_ms":sum(r["latency_ms"] for r in group)/len(group)}
    report = {"metrics": metrics,"details":details,"completion_sha256":file_sha256(args.output/"completion.json"),
              "limits":"public development comparison; unknown training/geographic overlap; no official score claim"}
    write_json(args.output/"metrics.json",report)
    print(json.dumps(metrics,indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    inference = commands.add_parser("predict")
    inference.add_argument("--manifest",type=Path,action="append",required=True)
    inference.add_argument("--model-dir",type=Path,required=True)
    inference.add_argument("--output",type=Path,required=True)
    inference.add_argument("--device",default="cpu")
    inference.add_argument("--threads",type=int,default=4)
    evaluation = commands.add_parser("score")
    evaluation.add_argument("--output",type=Path,required=True)
    evaluation.add_argument("--annotations",type=Path,required=True)
    evaluation.add_argument("--mask-dir",type=Path)
    args = parser.parse_args()
    (predict if args.command=="predict" else score)(args)


if __name__ == "__main__":
    main()
