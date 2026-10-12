"""Measure submission-oriented inference efficiency for scene classification."""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import ConcatDataset, Subset

from tasks.classification.data import SceneDataModule
from tasks.classification.inference import SceneClassificationPredictor
from tasks.evaluation_resources import (
    environment_summary,
    file_size_mb,
    latency_summary,
    model_parameters,
    peak_gpu_memory_mb,
    process_rss_mb,
    reset_peak_gpu_memory,
    synchronize,
)
from ultralytics.utils.torch_utils import get_flops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark batch-1 scene-classification inference efficiency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="runs/classification/vrsbench_scene/weights/best.pt")
    parser.add_argument("--pretrained-weights", default="pretrain_model/yolov12m.pt")
    parser.add_argument("--data-dir", default="data/VRSBench_scene")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--model", default="", help="Optional classification model-YAML override.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Used when data-dir has no explicit split folders.")
    parser.add_argument("--seed", type=int, default=42, help="Used for the deterministic stratified fallback split.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--warmup-samples", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0, help="0 benchmarks every image in the split.")
    parser.add_argument("--save-dir", default="runs/efficiency/classification")
    return parser


def _dataset_images(dataset: Any) -> list[Path]:
    if isinstance(dataset, Subset):
        source = _dataset_images(dataset.dataset)
        return [source[int(index)] for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [path for child in dataset.datasets for path in _dataset_images(child)]
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError(f"Unsupported classification benchmark dataset: {type(dataset).__name__}")
    return [Path(path).expanduser().resolve() for path, _ in samples]


def _collect_images(
    data_dir: str | Path,
    split: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[str]]:
    loader, classes = SceneDataModule(
        data_dir,
        imgsz=16,
        batch=1,
        workers=0,
        val_ratio=val_ratio,
        seed=seed,
        sampling="none",
    ).build_eval(split)
    images = _dataset_images(loader.dataset)
    if not images:
        raise ValueError(f"No benchmark images found for classification split: {split}")
    return images, classes


def _model_file_report(checkpoint: Path, pretrained: Path) -> dict[str, Any]:
    checkpoint_mb = file_size_mb(checkpoint)
    pretrained_mb = file_size_mb(pretrained)
    complete_mb = None if checkpoint_mb is None or pretrained_mb is None else checkpoint_mb + pretrained_mb
    return {
        "classification_checkpoint_mb": checkpoint_mb,
        "backbone_pretrained_mb": pretrained_mb,
        "complete_system_mb": complete_mb,
        "scope": "Head checkpoint plus the separately required YOLOv12m pretrained Backbone/Neck file.",
    }


def _save_outputs(save_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "efficiency.json"
    csv_path = save_dir / "sample_latencies.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "image", "latency_ms"))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.imgsz <= 0 or args.topk <= 0 or args.warmup_samples < 0 or args.max_samples < 0:
        raise ValueError("imgsz/topk must be positive and warmup/max-samples must be non-negative.")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    pretrained = Path(args.pretrained_weights).expanduser().resolve()
    images, dataset_classes = _collect_images(args.data_dir, args.split, args.val_ratio, args.seed)
    selected = images if args.max_samples == 0 else images[: args.max_samples]
    if not selected:
        raise ValueError("No samples remain after applying --max-samples.")

    load_started = time.perf_counter()
    predictor = SceneClassificationPredictor(
        checkpoint=checkpoint,
        pretrained_weights=pretrained,
        device=args.device,
        imgsz=args.imgsz,
        model_yaml=args.model,
    )
    if dataset_classes != predictor.class_names:
        raise ValueError("Benchmark dataset class order does not match the classification checkpoint.")
    synchronize(predictor.device)
    model_load_ms = (time.perf_counter() - load_started) * 1000.0

    synchronize(predictor.device)
    first_started = time.perf_counter()
    predictor.predict(selected[0], topk=args.topk)
    synchronize(predictor.device)
    first_request_ms = (time.perf_counter() - first_started) * 1000.0

    for image_path in islice(cycle(selected), args.warmup_samples):
        predictor.predict(image_path, topk=args.topk)
    synchronize(predictor.device)

    try:
        gflops = float(get_flops(predictor.network.model, imgsz=args.imgsz))
        gflops_error = None
    except Exception as exc:  # THOP support varies across installed versions.
        gflops = None
        gflops_error = f"{type(exc).__name__}: {exc}"

    reset_peak_gpu_memory(predictor.device)
    peak_rss = process_rss_mb()
    latencies_ms: list[float] = []
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(selected, start=1):
        synchronize(predictor.device)
        started = time.perf_counter()
        predictor.predict(image_path, topk=args.topk)
        synchronize(predictor.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(latency_ms)
        peak_rss = max(peak_rss, process_rss_mb())
        rows.append({"sample_index": index, "image": str(image_path), "latency_ms": latency_ms})

    latency = latency_summary(latencies_ms)
    artifacts = _model_file_report(checkpoint, pretrained)
    peak_gpu_mb = peak_gpu_memory_mb(predictor.device)
    report = {
        "task": "scene_classification",
        "benchmark": {
            "batch_size": 1,
            "input_size": [args.imgsz, args.imgsz],
            "precision": "fp32",
            "split": args.split,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "available_samples": len(images),
            "measured_samples": len(selected),
            "warmup_samples": args.warmup_samples,
            "failed_samples": 0,
            "model_load_ms": model_load_ms,
            "first_request_ms": first_request_ms,
            "latency": latency,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "peak_process_rss_mb": peak_rss,
            "scope": "image decode + RGB conversion + resize/normalize + host-to-device transfer + classification + softmax + Top-K; model loading excluded",
        },
        "parameters": model_parameters(predictor.network),
        "gflops": {
            "complete_task_model": gflops,
            "method": "THOP multiply-add count multiplied by two at batch=1 and the configured square input size.",
            "error": gflops_error,
        },
        "model_files": artifacts,
        "table_metrics": {
            "gflops": gflops,
            "model_file_size_mb": artifacts["complete_system_mb"],
            "peak_gpu_memory_gb": None if peak_gpu_mb is None else peak_gpu_mb / 1024.0,
            "peak_cpu_memory_gb": peak_rss / 1024.0,
            "mean_latency_ms": latency["mean_ms"],
            "p95_latency_ms": latency["p95_ms"],
            "ttft_ms": None,
            "decoding_speed": None,
        },
        "not_applicable": {
            "ttft_ms": "Scene classification is non-autoregressive and emits no text token stream.",
            "decoding_speed": "Scene classification has no token decoding stage.",
        },
        "environment": environment_summary(predictor.device),
    }
    json_path, csv_path = _save_outputs(Path(args.save_dir).expanduser().resolve(), report, rows)
    print(json.dumps(report["table_metrics"], ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"Per-sample CSV: {csv_path}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
