"""Measure submission-oriented inference efficiency for text-guided object counting."""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Sequence

import cv2
import torch
import torch.nn as nn
import thop

from tasks.counting.data import VRSCountingDataset
from tasks.counting.inference import CountingPredictor
from tasks.evaluation_resources import (
    environment_summary,
    file_size_mb,
    latency_summary,
    model_parameters,
    peak_gpu_memory_mb,
    process_rss_mb,
    reset_peak_gpu_memory,
    resolve_openclip_weight_file,
    synchronize,
)
from ultralytics.utils.ops import scale_boxes


class _CountingForwardForFlops(nn.Module):
    def __init__(self, model: nn.Module, text_vector: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("text_vector", text_vector)
        self.register_buffer("token_mask", torch.ones((1, text_vector.shape[1]), dtype=torch.bool, device=text_vector.device))

    def forward(self, image: torch.Tensor) -> Any:
        output = self.model(image, txt_vec=self.text_vector, txt_token_mask=self.token_mask)
        return output[0] if isinstance(output, tuple) else output


class _PooledOpenCLIPTextForFlops(nn.Module):
    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.clip_model = clip_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_text(tokens)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark batch-1 text-guided object-counting inference efficiency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt")
    parser.add_argument("--voc-root", default="data/VRSBench")
    parser.add_argument("--split-file", default="test.txt")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-model-name", default="ViT-L-14")
    parser.add_argument("--text-pretrained", default="openai")
    parser.add_argument("--prompt-template", default="a satellite remote sensing photo of {class_name}")
    parser.add_argument("--openclip-weight-file", default="", help="Optional cache-file override; the standard OpenAI CLIP cache path is detected automatically.")
    parser.add_argument("--conf-thres", type=float, default=0.15)
    parser.add_argument("--iou-thres", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=0, help="0 benchmarks every positive image-class query pair.")
    parser.add_argument("--clear-text-cache-each-query", action="store_true", help="Measure online OpenCLIP encoding for every query instead of the deployed prompt cache.")
    parser.add_argument("--save-dir", default="runs/efficiency/counting")
    return parser


def _query_pairs(dataset: VRSCountingDataset) -> list[tuple[str, Path, str]]:
    pairs = [
        (sample.image_id, sample.image_path, class_name)
        for sample in dataset
        for class_name in sample.counts
    ]
    if not pairs:
        raise ValueError("No positive image-class query pairs were found for the counting benchmark.")
    return pairs


def _predict_without_visualization(predictor: CountingPredictor, image_path: Path, target_class: str, args: argparse.Namespace) -> int:
    runtime = predictor.detector
    if args.clear_text_cache_each_query:
        runtime.detector.prompt_encoder.cache.clear()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to decode benchmark image: {image_path}")
    image_tensor, _, _ = runtime.preprocessor(image)
    result = runtime.detector.detect(
        image_tensor,
        target_class,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
    )
    boxes = result["boxes"].clone()
    if len(boxes) > 0:
        scale_boxes(image_tensor.shape[2:], boxes, image.shape[:2])
    return int(len(boxes))


def _profile_gflops(predictor: CountingPredictor, prompt: str, imgsz: int) -> dict[str, Any]:
    runtime = predictor.detector.detector
    result: dict[str, Any] = {"task_model": None, "openclip_text_encoder": None, "complete_path_estimate": None, "errors": []}
    try:
        text_vector = runtime.prompt_encoder.encode(prompt)
        wrapper = _CountingForwardForFlops(runtime.model, text_vector).eval()
        image = torch.zeros((1, 3, imgsz, imgsz), device=runtime.device)
        macs, _ = thop.profile(wrapper, inputs=(image,), verbose=False)
        result["task_model"] = float(macs * 2.0 / 1e9)
    except Exception as exc:
        result["errors"].append(f"task_model: {type(exc).__name__}: {exc}")
    try:
        natural = runtime.prompt_encoder.clean_prompt(prompt)
        rendered = runtime.prompt_encoder.prompt_template.format(class_name=natural)
        tokens = runtime.prompt_encoder.tokenizer([rendered]).to(runtime.device)
        wrapper = _PooledOpenCLIPTextForFlops(runtime.prompt_encoder.clip_model).eval()
        macs, _ = thop.profile(wrapper, inputs=(tokens,), verbose=False)
        result["openclip_text_encoder"] = float(macs * 2.0 / 1e9)
    except Exception as exc:
        result["errors"].append(f"openclip_text_encoder: {type(exc).__name__}: {exc}")
    if result["task_model"] is not None and result["openclip_text_encoder"] is not None:
        result["complete_path_estimate"] = result["task_model"] + result["openclip_text_encoder"]
    result["method"] = "THOP MACs multiplied by two; complete estimate includes the online OpenCLIP text path and task model, but excludes image preprocessing and NMS."
    return result


def _model_files(checkpoint: Path, openclip_weight_file: str, model_name: str, pretrained: str) -> dict[str, Any]:
    checkpoint_mb = file_size_mb(checkpoint)
    resolved_openclip = resolve_openclip_weight_file(openclip_weight_file, model_name, pretrained)
    openclip_mb = file_size_mb(resolved_openclip) if resolved_openclip is not None else None
    complete_mb = None if checkpoint_mb is None or openclip_mb is None else checkpoint_mb + openclip_mb
    return {
        "counting_checkpoint_mb": checkpoint_mb,
        "openclip_weight_file": str(resolved_openclip) if resolved_openclip is not None else None,
        "openclip_weight_file_mb": openclip_mb,
        "complete_system_mb": complete_mb,
        "scope": "Complete storage includes the counting checkpoint and cached OpenCLIP weight file. The standard OpenAI cache is auto-detected; use --openclip-weight-file for a custom cache.",
    }


def _save_outputs(save_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "efficiency.json"
    csv_path = save_dir / "query_latencies.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("query_index", "image_id", "image", "target_class", "pred_count", "latency_ms"))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.imgsz <= 0 or args.warmup_queries < 0 or args.max_queries < 0 or args.max_det <= 0:
        raise ValueError("imgsz/max-det must be positive and warmup/max-queries must be non-negative.")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    dataset = VRSCountingDataset(args.voc_root, split_file=args.split_file)
    pairs = _query_pairs(dataset)
    selected = pairs if args.max_queries == 0 else pairs[: args.max_queries]
    if not selected:
        raise ValueError("No query pairs remain after applying --max-queries.")

    load_started = time.perf_counter()
    predictor = CountingPredictor(
        checkpoint=checkpoint,
        device=args.device,
        imgsz=args.imgsz,
        text_model_name=args.text_model_name,
        text_pretrained=args.text_pretrained,
        prompt_template=args.prompt_template,
    )
    device = predictor.detector.device
    synchronize(device)
    model_load_ms = (time.perf_counter() - load_started) * 1000.0

    first_id, first_image, first_prompt = selected[0]
    del first_id
    synchronize(device)
    first_started = time.perf_counter()
    _predict_without_visualization(predictor, first_image, first_prompt, args)
    synchronize(device)
    first_request_ms = (time.perf_counter() - first_started) * 1000.0

    for _, image_path, prompt in islice(cycle(selected), args.warmup_queries):
        _predict_without_visualization(predictor, image_path, prompt, args)
    synchronize(device)

    gflops = _profile_gflops(predictor, first_prompt, args.imgsz)
    reset_peak_gpu_memory(device)
    peak_rss = process_rss_mb()
    latencies_ms: list[float] = []
    rows: list[dict[str, Any]] = []
    failed_queries = 0
    for index, (image_id, image_path, prompt) in enumerate(selected, start=1):
        try:
            synchronize(device)
            started = time.perf_counter()
            pred_count = _predict_without_visualization(predictor, image_path, prompt, args)
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            failed_queries += 1
            rows.append({"query_index": index, "image_id": image_id, "image": str(image_path), "target_class": prompt, "pred_count": "", "latency_ms": ""})
            print(f"warning: query {image_id}/{prompt} failed: {type(exc).__name__}: {exc}")
            continue
        latencies_ms.append(latency_ms)
        peak_rss = max(peak_rss, process_rss_mb())
        rows.append({"query_index": index, "image_id": image_id, "image": str(image_path), "target_class": prompt, "pred_count": pred_count, "latency_ms": latency_ms})

    if not latencies_ms:
        raise RuntimeError("Every counting benchmark query failed.")
    latency = latency_summary(latencies_ms)
    task_parameters = model_parameters(predictor.detector.detector.model)
    text_parameters = model_parameters(predictor.detector.detector.prompt_encoder.clip_model)
    model_files = _model_files(
        checkpoint,
        args.openclip_weight_file,
        args.text_model_name,
        args.text_pretrained,
    )
    peak_gpu_mb = peak_gpu_memory_mb(device)
    report = {
        "task": "object_counting",
        "benchmark": {
            "batch_size": 1,
            "input_size": [args.imgsz, args.imgsz],
            "precision": str(next(predictor.detector.detector.model.parameters()).dtype).removeprefix("torch."),
            "split_file": args.split_file,
            "available_query_pairs": len(pairs),
            "measured_query_pairs": len(latencies_ms),
            "failed_queries": failed_queries,
            "warmup_queries": args.warmup_queries,
            "text_cache_strategy": "cleared before every query" if args.clear_text_cache_each_query else "deployed in-process prompt cache enabled",
            "model_load_ms": model_load_ms,
            "first_request_ms": first_request_ms,
            "latency": latency,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "peak_process_rss_mb": peak_rss,
            "scope": "image decode + 800-square letterbox/normalize + host-to-device transfer + prompt encoding/cache + detector + class-agnostic NMS + native-coordinate restoration + counting; visualization and file saving excluded",
        },
        "parameters": {
            "task_model": task_parameters,
            "openclip_loaded_model": text_parameters,
            "complete_system_total": task_parameters["total"] + text_parameters["total"],
            "complete_system_loaded_parameter_memory_mb": task_parameters["loaded_parameter_memory_mb"] + text_parameters["loaded_parameter_memory_mb"],
        },
        "gflops": gflops,
        "model_files": model_files,
        "table_metrics": {
            "gflops": gflops["complete_path_estimate"],
            "model_file_size_mb": model_files["complete_system_mb"],
            "peak_gpu_memory_gb": None if peak_gpu_mb is None else peak_gpu_mb / 1024.0,
            "peak_cpu_memory_gb": peak_rss / 1024.0,
            "mean_latency_ms": latency["mean_ms"],
            "p95_latency_ms": latency["p95_ms"],
            "ttft_ms": None,
            "decoding_speed": None,
        },
        "not_applicable": {
            "ttft_ms": "Object counting is non-autoregressive and emits a scalar count rather than text tokens.",
            "decoding_speed": "Object counting has no token decoding stage.",
        },
        "environment": environment_summary(device),
    }
    json_path, csv_path = _save_outputs(Path(args.save_dir).expanduser().resolve(), report, rows)
    print(json.dumps(report["table_metrics"], ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"Per-query CSV: {csv_path}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
