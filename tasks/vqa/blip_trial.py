"""Isolated public BLIP-VQA comparison; never imported by the API or training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

from .config import SUPPORTED, resolve_path, sha256, write_json
from .data import normalize_answer, read_rows


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row["sample_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate sample IDs")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_balanced(rows: list[dict], per_type: int, seed: int) -> list[dict]:
    """Selection depends on type and stable ID, never answers or predictions."""
    if per_type < 1:
        raise ValueError("per_type must be positive")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["question_type"]].append(row)
    selected = []
    for kind in SUPPORTED:
        if len(grouped[kind]) < per_type:
            raise ValueError(f"Insufficient public rows: {kind}")
        ranked = sorted(grouped[kind], key=lambda row: (
            hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest(),
            row["sample_id"],
        ))
        selected.extend(ranked[:per_type])
    return selected


def prepare(args) -> None:
    from PIL import Image

    source, out = resolve_path(args.jsonl), resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("inputs.jsonl", "gold.jsonl", "protocol.json")):
        raise FileExistsError("Trial selection already exists; use another output directory")
    rows = read_rows(source)
    if any(row.get("source") != "VRSBench" or row.get("split") != "val" for row in rows):
        raise ValueError("This protocol requires original public VRSBench val")
    selected = select_balanced(rows, args.per_type, args.seed)
    inputs, gold = [], []
    seen_images, decode_differences = {}, []
    for row in selected:
        path = resolve_path(row["image_path"])
        if path not in seen_images:
            with Image.open(path) as original:
                image = original.convert("RGB")
                seen_images[path] = dict(
                    image_sha256=sha256(path),
                    pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    width=image.width, height=image.height,
                )
            if seen_images[path]["pixel_sha256"] != row["pixel_sha256"]:
                decode_differences.append(dict(image_id=row["image_id"],
                    declared_pixel_sha256=row["pixel_sha256"],
                    local_pixel_sha256=seen_images[path]["pixel_sha256"]))
        meta = seen_images[path]
        for key in ("image_sha256", "width", "height"):
            if meta[key] != row[key]:
                raise ValueError(f"Public image metadata mismatch: {key}")
        inputs.append(dict(sample_id=row["sample_id"], image_id=row["image_id"],
                           image_path=str(path), question=row["question"], **meta))
        gold.append(dict(sample_id=row["sample_id"], question_type=row["question_type"],
                         normalized_answer=normalize_answer(row["answer"])))
    write_jsonl(out / "inputs.jsonl", inputs)
    write_jsonl(out / "gold.jsonl", gold)
    write_json(out / "protocol.json", dict(
        public_source="VRSBench original val; development selection, not independent test",
        source_jsonl_sha256=sha256(source), seed=args.seed, per_type=args.per_type,
        selection="per-type lowest SHA256(seed:sample_id), ties ordered by sample_id",
        n=len(inputs), images=len(seen_images), gold_counts=dict(Counter(r["question_type"] for r in gold)),
        inputs_sha256=sha256(out / "inputs.jsonl"), gold_sha256=sha256(out / "gold.jsonl"),
        scoring="project normalize_answer + exact match; failures count wrong; gold read only after both completions",
        blip=dict(model_id="Salesforce/blip-vqa-base", revision=args.revision,
                  dtype="float32", num_beams=3, do_sample=False, max_new_tokens=20,
                  full_image=True, fine_tuning=False, question_rewrite=False),
        execution=dict(blip_device=args.blip_device, native_device=args.native_device,
                       threads=args.threads, batch=1, warmup="first selected input, unscored timing"),
        latency_note="Devices may differ; report device explicitly, no cross-device speed claim",
        frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        local_decode_differences=decode_differences,
        decode_note="Original file hashes and dimensions must match; any decoded-pixel difference from delivery is recorded. Both models receive the same local RGB decode.",
        source_sha256={str(path): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name("data.py"),
            Path(__file__).with_name("inference.py"), Path(__file__).with_name("model.py"),
            Path(__file__).with_name("qformer_core.py"), Path(__file__).with_name("config.py"),
        )},
    ))
    print(json.dumps(dict(n=len(inputs), images=len(seen_images), inputs_sha256=sha256(out / "inputs.jsonl"))), flush=True)


def predict(args) -> None:
    import resource
    import torch
    from PIL import Image

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261013)
    out = resolve_path(args.out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if sha256(out / "inputs.jsonl") != protocol["inputs_sha256"]:
        raise ValueError("Frozen inputs changed")
    if any(sha256(Path(path)) != digest for path, digest in protocol["source_sha256"].items()):
        raise ValueError("Frozen VQA source changed")
    inputs = read_jsonl(out / "inputs.jsonl")
    allowed_fields = {"sample_id", "image_id", "image_path", "question", "image_sha256", "pixel_sha256", "width", "height"}
    if len(inputs) != protocol["n"] or any(set(row) != allowed_fields for row in inputs):
        raise ValueError("Invalid prediction-only input manifest")
    device_name = protocol["execution"][f"{args.backend}_device"]
    if args.threads != protocol["execution"]["threads"]:
        raise ValueError("Thread setting differs from frozen protocol")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA device unavailable")
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if free < 3 * 1024**3:
            raise RuntimeError("Insufficient free VRAM; no shared service is stopped")
    destination = out / f"{args.backend}_predictions.jsonl"
    if destination.exists() or (out / f"{args.backend}_completion.json").exists():
        raise FileExistsError("Predictions already exist; do not overwrite trial evidence")
    load_start = time.perf_counter()
    source = dict()
    if args.backend == "blip":
        import transformers
        from transformers import BlipForQuestionAnswering, BlipProcessor

        root = resolve_path(args.model_dir)
        receipt = json.loads((root / "source_receipt.json").read_text(encoding="utf-8-sig"))
        if receipt["model_id"] != protocol["blip"]["model_id"] or receipt["revision"] != protocol["blip"]["revision"]:
            raise ValueError("Model provenance differs from frozen protocol")
        for entry in receipt["files"]:
            if sha256(root / entry["name"]) != entry["sha256"]:
                raise ValueError("Model asset hash mismatch")
        processor = BlipProcessor.from_pretrained(root, local_files_only=True)
        model, loading = BlipForQuestionAnswering.from_pretrained(
            root, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float32, output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete BLIP loading: {loading}")
        model = model.eval().to(device)
        source = dict(receipt=receipt, loading_info=loading, transformers=transformers.__version__,
                      parameters=sum(p.numel() for p in model.parameters()))

        @torch.inference_mode()
        def infer(image, question):
            encoded = processor(images=image, text=question, return_tensors="pt", truncation=False)
            if encoded["input_ids"].shape[1] > model.config.text_config.max_position_embeddings:
                return dict(status="unsupported", answer_text=None, reason="text_context_exceeded")
            encoded = encoded.to(device)
            tokens = model.generate(**encoded, num_beams=3, do_sample=False, max_new_tokens=20)
            answer = processor.decode(tokens[0], skip_special_tokens=True).strip()
            return dict(status="ok" if answer else "error", answer_text=answer or None,
                        reason=None if answer else "empty_generation")
    else:
        from .inference import VQAPredictor

        predictor = VQAPredictor(args.bundle_dir, device_name, dict(
            visual_backbone=args.visual_backbone, text_encoder=args.text_encoder,
        ))
        source = dict(head_sha256=predictor.manifest["checkpoint"]["sha256"],
                      external_assets=predictor.manifest["external_assets"])
        infer = predictor.predict
    load_seconds = time.perf_counter() - load_start

    def checked_image(row):
        if sha256(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError("Frozen public image changed")
        with Image.open(row["image_path"]) as original:
            image = original.convert("RGB")
        if hashlib.sha256(image.tobytes()).hexdigest() != row["pixel_sha256"] or image.size != (row["width"], row["height"]):
            raise ValueError("Frozen local RGB decode changed")
        return image

    infer(checked_image(inputs[0]), inputs[0]["question"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with destination.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(inputs, 1):
            image = checked_image(row)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            try:
                result = infer(image, row["question"])
            except (RuntimeError, ValueError) as error:
                result = dict(status="error", answer_text=None, reason=type(error).__name__)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - before
            stream.write(json.dumps(dict(sample_id=row["sample_id"], result=result, seconds=elapsed), ensure_ascii=False) + "\n")
            stream.flush()
            if index % 10 == 0:
                print(f"{args.backend} {index}/{len(inputs)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    if any(sha256(Path(path)) != digest for path, digest in protocol["source_sha256"].items()):
        raise ValueError("VQA source changed during prediction")
    write_json(out / f"{args.backend}_completion.json", dict(
        n=len(inputs), backend=args.backend, device=str(device), dtype="float32",
        torch=torch.__version__, python=platform.python_version(), threads=args.threads,
        load_seconds=load_seconds, prediction_wall_seconds=time.perf_counter()-start,
        process_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2 if device.type == "cuda" else None,
        inputs_sha256=protocol["inputs_sha256"], predictions_sha256=sha256(destination),
        source=source, completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    ))


def score_predictions(inputs: list[dict], gold: list[dict], predictions: dict[str, list[dict]]) -> dict:
    expected = [row["sample_id"] for row in inputs]
    if [row["sample_id"] for row in gold] != expected:
        raise ValueError("Gold IDs/order differ from frozen inputs")
    metrics, correctness = {}, {}
    for name, rows in predictions.items():
        if [row["sample_id"] for row in rows] != expected:
            raise ValueError("Prediction IDs/order/incomplete denominator")
        correct = [bool(row["result"]["status"] == "ok" and normalize_answer(row["result"].get("answer_text", "")) == target["normalized_answer"])
                   for row, target in zip(rows, gold)]
        correctness[name] = correct
        per_type = {}
        for kind in SUPPORTED:
            indices = [i for i, target in enumerate(gold) if target["question_type"] == kind]
            if not indices:
                raise ValueError("Missing type in frozen trial")
            per_type[kind] = dict(n=len(indices), correct=sum(correct[i] for i in indices),
                                  accuracy=sum(correct[i] for i in indices)/len(indices))
        seconds = sorted(row["seconds"] for row in rows)
        metrics[name] = dict(n=len(rows), correct=sum(correct), accuracy=sum(correct)/len(rows),
                             macro_accuracy=sum(v["accuracy"] for v in per_type.values())/len(per_type),
                             failures=sum(row["result"]["status"] != "ok" for row in rows), per_type=per_type,
                             latency=dict(mean_seconds=sum(seconds)/len(seconds),
                                          p50_seconds=seconds[len(seconds)//2],
                                          p95_seconds=seconds[min(len(seconds)-1, int(.95*len(seconds)))]))
    native, blip = correctness["native"], correctness["blip"]
    paired = dict(gain=sum(b and not a for a,b in zip(native,blip)),
                  loss=sum(a and not b for a,b in zip(native,blip)),
                  both_correct=sum(a and b for a,b in zip(native,blip)),
                  both_wrong=sum(not a and not b for a,b in zip(native,blip)))
    return dict(metrics=metrics, paired_blip_vs_native=paired,
                accuracy_note="Balanced 120-row public development slice; not full val, independent test or official score")


def score(args) -> None:
    out = resolve_path(args.out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    predictions, completions = {}, {}
    for name in ("native", "blip"):
        completion = json.loads((out / f"{name}_completion.json").read_text(encoding="utf-8"))
        path = out / f"{name}_predictions.jsonl"
        if completion["inputs_sha256"] != protocol["inputs_sha256"] or completion["n"] != protocol["n"] or sha256(path) != completion["predictions_sha256"]:
            raise ValueError("Incomplete or changed prediction evidence")
        completions[name] = completion
        predictions[name] = read_jsonl(path)
    # No gold is opened until both complete predictions have been verified.
    if sha256(out / "inputs.jsonl") != protocol["inputs_sha256"] or sha256(out / "gold.jsonl") != protocol["gold_sha256"]:
        raise ValueError("Frozen selection/gold changed")
    report = score_predictions(read_jsonl(out / "inputs.jsonl"), read_jsonl(out / "gold.jsonl"), predictions)
    report.update(protocol=protocol, resources=completions)
    write_json(out / "comparison.json", report)
    print(json.dumps({key: report[key] for key in ("metrics", "paired_blip_vs_native")}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("prepare")
    selection.add_argument("--jsonl", required=True)
    selection.add_argument("--out", required=True)
    selection.add_argument("--per-type", type=int, default=10)
    selection.add_argument("--seed", type=int, default=20261013)
    selection.add_argument("--revision", required=True)
    selection.add_argument("--blip-device", default="cuda:0")
    selection.add_argument("--native-device", default="cpu")
    selection.add_argument("--threads", type=int, default=2)
    inference = sub.add_parser("predict")
    inference.add_argument("--out", required=True)
    inference.add_argument("--backend", choices=("native", "blip"), required=True)
    inference.add_argument("--model-dir")
    inference.add_argument("--bundle-dir")
    inference.add_argument("--visual-backbone")
    inference.add_argument("--text-encoder")
    inference.add_argument("--threads", type=int, default=2)
    scoring = sub.add_parser("score")
    scoring.add_argument("--out", required=True)
    args = parser.parse_args()
    dict(prepare=prepare, predict=predict, score=score)[args.command](args)


if __name__ == "__main__":
    main()
