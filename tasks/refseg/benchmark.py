"""Measure submission-oriented inference efficiency for referring segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import thop
from ultralytics.utils import yaml_load

from dataset.rrsisd_refseg_dataset import clean_rrsisd_rows
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
from tasks.refseg.inference import RefSegPredictor
from tasks.refseg.runtime import PROJECT_ROOT, resolve_data_root, resolve_existing_path, resolve_split_path


class _RefSegForwardForFlops(nn.Module):
    def __init__(self, model: nn.Module, text_features: torch.Tensor, token_mask: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)
        self.register_buffer("token_mask", token_mask)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image, text_embedding=self.text_features, text_token_mask=self.token_mask)
        return output[0] if isinstance(output, (list, tuple)) else output


class _TokenOpenCLIPTextForFlops(nn.Module):
    def __init__(self, encoder: Any) -> None:
        super().__init__()
        self.encoder = encoder.model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.encoder.token_embedding(tokens)
        positional = getattr(self.encoder, "positional_embedding", None)
        if isinstance(positional, torch.Tensor):
            x = x + positional[: x.shape[1]]
        x = x.permute(1, 0, 2)
        attention_mask = getattr(self.encoder, "attn_mask", None)
        try:
            x = self.encoder.transformer(x, attn_mask=attention_mask)
        except TypeError:
            x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)
        if getattr(self.encoder, "ln_final", None) is not None:
            x = self.encoder.ln_final(x)
        projection = getattr(self.encoder, "text_projection", None)
        return x @ projection if isinstance(projection, torch.Tensor) else x


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark batch-1 referring-segmentation inference efficiency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="runs/semseg/srp_yolov12m_axis/weights/best_raw.pt")
    parser.add_argument("--data", default="pre_datasets/RRSIS-D_refseg/data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default="")
    parser.add_argument("--imgsz", type=int, default=None, help="Defaults to the checkpoint value (512 for the formal model).")
    parser.add_argument("--threshold", type=float, default=None, help="Defaults to the validation-selected checkpoint threshold.")
    parser.add_argument("--text-model-name", default="")
    parser.add_argument("--text-pretrained", default="")
    parser.add_argument("--text-precision", default="")
    parser.add_argument("--openclip-weight-file", default="", help="Optional cache-file override; the standard OpenAI CLIP cache path is detected automatically.")
    parser.add_argument("--warmup-samples", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0, help="0 benchmarks the complete cleaned split.")
    parser.add_argument("--empty-mask-policy", choices=("drop", "error", "keep"), default="drop")
    parser.add_argument("--save-dir", default="runs/efficiency/refseg")
    return parser


def _load_rows(data_path: str | Path, split: str, empty_mask_policy: str) -> list[dict[str, Any]]:
    resolved_data = Path(data_path).expanduser().resolve()
    if not resolved_data.is_file():
        raise FileNotFoundError(f"RefSeg data YAML not found: {resolved_data}")
    data = dict(yaml_load(resolved_data))
    data["_yaml_dir"] = str(resolved_data.parent)
    split_path = resolve_split_path(data, split)
    if not split_path.is_file():
        raise FileNotFoundError(f"RefSeg split JSONL not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows, _ = clean_rrsisd_rows(rows, empty_mask_policy=empty_mask_policy, context=str(split_path))
    data_root = resolve_data_root(data)
    for row in rows:
        configured = Path(str(row["image"])).expanduser()
        resolved = resolve_existing_path(configured)
        if resolved is None and not configured.is_absolute():
            resolved = resolve_existing_path(PROJECT_ROOT / configured)
        if resolved is None and not configured.is_absolute():
            resolved = resolve_existing_path(data_root / configured)
        if resolved is None or not resolved.is_file():
            raise FileNotFoundError(f"RefSeg benchmark image not found: {row['image']}")
        row["_resolved_image"] = str(resolved.resolve())
    if not rows:
        raise ValueError(f"No RefSeg samples available in split: {split}")
    return rows


def _profile_gflops(predictor: RefSegPredictor, prompt: str) -> dict[str, Any]:
    result: dict[str, Any] = {"task_model": None, "openclip_text_encoder": None, "complete_path_estimate": None, "errors": []}
    try:
        features, token_mask = predictor.text_encoder.encode([prompt], batch_size=1, return_tokens=True)
        features = features.to(device=predictor.device, dtype=torch.float32)
        token_mask = token_mask.to(device=predictor.device, dtype=torch.bool)
        wrapper = _RefSegForwardForFlops(predictor.model, features, token_mask)
        image = torch.zeros((1, 3, predictor.imgsz, predictor.imgsz), device=predictor.device)
        macs, _ = thop.profile(wrapper, inputs=(image,), verbose=False)
        result["task_model"] = float(macs * 2.0 / 1e9)
    except Exception as exc:
        result["errors"].append(f"task_model: {type(exc).__name__}: {exc}")
    try:
        tokens = predictor.text_encoder.tokenizer([prompt]).to(predictor.device)
        wrapper = _TokenOpenCLIPTextForFlops(predictor.text_encoder)
        macs, _ = thop.profile(wrapper, inputs=(tokens,), verbose=False)
        result["openclip_text_encoder"] = float(macs * 2.0 / 1e9)
    except Exception as exc:
        result["errors"].append(f"openclip_text_encoder: {type(exc).__name__}: {exc}")
    if result["task_model"] is not None and result["openclip_text_encoder"] is not None:
        result["complete_path_estimate"] = result["task_model"] + result["openclip_text_encoder"]
    result["method"] = "THOP MACs multiplied by two; complete estimate includes online token-level OpenCLIP encoding and the RefSeg model, but excludes image/mask postprocessing."
    return result


def _model_files(checkpoint: Path, openclip_weight_file: str, model_name: str, pretrained: str) -> dict[str, Any]:
    checkpoint_mb = file_size_mb(checkpoint)
    resolved_openclip = resolve_openclip_weight_file(openclip_weight_file, model_name, pretrained)
    openclip_mb = file_size_mb(resolved_openclip) if resolved_openclip is not None else None
    complete_mb = None if checkpoint_mb is None or openclip_mb is None else checkpoint_mb + openclip_mb
    return {
        "refseg_checkpoint_mb": checkpoint_mb,
        "openclip_weight_file": str(resolved_openclip) if resolved_openclip is not None else None,
        "openclip_weight_file_mb": openclip_mb,
        "complete_system_mb": complete_mb,
        "scope": "Complete storage includes best_raw.pt and the cached OpenCLIP weight file. The standard OpenAI cache is auto-detected; use --openclip-weight-file for a custom cache.",
    }


def _save_outputs(save_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "efficiency.json"
    csv_path = save_dir / "sample_latencies.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "sample_id", "image", "prompt", "latency_ms"))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup_samples < 0 or args.max_samples < 0:
        raise ValueError("warmup-samples and max-samples must be non-negative.")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    rows = _load_rows(args.data, args.split, args.empty_mask_policy)
    selected = rows if args.max_samples == 0 else rows[: args.max_samples]
    if not selected:
        raise ValueError("No samples remain after applying --max-samples.")

    load_started = time.perf_counter()
    predictor = RefSegPredictor(
        checkpoint=checkpoint,
        device=args.device,
        model_yaml=args.model,
        imgsz=args.imgsz,
        threshold=args.threshold,
        text_model_name=args.text_model_name,
        text_pretrained=args.text_pretrained,
        text_precision=args.text_precision,
    )
    synchronize(predictor.device)
    model_load_ms = (time.perf_counter() - load_started) * 1000.0

    first = selected[0]
    synchronize(predictor.device)
    first_started = time.perf_counter()
    predictor.predict(first["_resolved_image"], str(first["text"]))
    synchronize(predictor.device)
    first_request_ms = (time.perf_counter() - first_started) * 1000.0

    for row in islice(cycle(selected), args.warmup_samples):
        predictor.predict(row["_resolved_image"], str(row["text"]))
    synchronize(predictor.device)

    gflops = _profile_gflops(predictor, str(first["text"]))
    reset_peak_gpu_memory(predictor.device)
    peak_rss = process_rss_mb()
    latencies_ms: list[float] = []
    output_rows: list[dict[str, Any]] = []
    failed_samples = 0
    for index, row in enumerate(selected, start=1):
        try:
            synchronize(predictor.device)
            started = time.perf_counter()
            predictor.predict(row["_resolved_image"], str(row["text"]))
            synchronize(predictor.device)
            latency_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            failed_samples += 1
            output_rows.append({"sample_index": index, "sample_id": row.get("id", ""), "image": row["_resolved_image"], "prompt": row.get("text", ""), "latency_ms": ""})
            print(f"warning: sample {row.get('id', index)} failed: {type(exc).__name__}: {exc}")
            continue
        latencies_ms.append(latency_ms)
        peak_rss = max(peak_rss, process_rss_mb())
        output_rows.append({"sample_index": index, "sample_id": row.get("id", ""), "image": row["_resolved_image"], "prompt": row.get("text", ""), "latency_ms": latency_ms})

    if not latencies_ms:
        raise RuntimeError("Every RefSeg benchmark sample failed.")
    latency = latency_summary(latencies_ms)
    task_parameters = model_parameters(predictor.model)
    text_parameters = model_parameters(predictor.text_encoder.model)
    model_files = _model_files(
        checkpoint,
        args.openclip_weight_file,
        predictor.text_model_name,
        predictor.text_pretrained,
    )
    peak_gpu_mb = peak_gpu_memory_mb(predictor.device)
    report = {
        "task": "referring_segmentation",
        "benchmark": {
            "batch_size": 1,
            "input_size": [predictor.imgsz, predictor.imgsz],
            "precision": {
                "task_model": str(next(predictor.model.parameters()).dtype).removeprefix("torch."),
                "openclip": predictor.text_precision,
            },
            "mask_threshold": predictor.threshold,
            "threshold_source": "validation-selected checkpoint value" if args.threshold is None else "explicit CLI override",
            "split": args.split,
            "available_samples": len(rows),
            "measured_samples": len(latencies_ms),
            "failed_samples": failed_samples,
            "warmup_samples": args.warmup_samples,
            "model_load_ms": model_load_ms,
            "first_request_ms": first_request_ms,
            "latency": latency,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "peak_process_rss_mb": peak_rss,
            "scope": "image decode + RGB conversion + resize/normalize + host-to-device transfer + online token-level OpenCLIP encoding + RefSeg forward + original-size mask restoration + CPU conversion + thresholding + overlay composition; file saving and model loading excluded",
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
            "ttft_ms": "Referring segmentation is non-autoregressive and emits a dense mask rather than text tokens.",
            "decoding_speed": "Referring segmentation has no token decoding stage.",
        },
        "environment": environment_summary(predictor.device),
    }
    json_path, csv_path = _save_outputs(Path(args.save_dir).expanduser().resolve(), report, output_rows)
    print(json.dumps(report["table_metrics"], ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"Per-sample CSV: {csv_path}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
