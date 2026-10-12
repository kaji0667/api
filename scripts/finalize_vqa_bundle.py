"""Seal a full Q-Former training run after complete cached validation.

No checkpoint conversion, model selection, private input or credential access.
This records immutable receipts required by the serving factory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seal(root: Path) -> dict:
    root = root.resolve()
    read = lambda relative: json.loads((root / relative).read_text(encoding="utf-8"))
    manifest = read("manifest.json")
    core = ["weights/vqa_head.pt", "config/train.json", "config/inference.json", "config/labels.json"]
    for relative in core:
        expected = manifest.get("file_sha256", {}).get(relative)
        if not isinstance(expected, str) or digest(root / relative) != expected:
            raise ValueError("Original trained weights/configuration changed or lost their binding")
    config, training, metrics, environment, labels_audit = (read(name) for name in (
        "config/train.json", "training_result.json", "evaluation/metrics.json",
        "environment.json", "labels_audit.json"))
    checkpoint_hash = manifest.get("checkpoint", {}).get("sha256")
    dataset_hash = config.get("val_jsonl_sha256")
    for value in (checkpoint_hash, dataset_hash):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Missing or invalid checkpoint/dataset SHA-256")
    epochs, best_epoch = training.get("epochs_completed"), training.get("best_epoch")
    score, selected_score = metrics.get("first_stage_macro_accuracy"), training.get("best_macro_accuracy")
    count = metrics.get("n")
    if (type(epochs) is not int or type(best_epoch) is not int or not 0 < best_epoch <= epochs
            or type(config.get("epochs")) is not int or epochs > config["epochs"]
            or type(count) is not int or count <= 0
            or not all(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1
                       for value in (score, selected_score, metrics.get("accuracy")))):
        raise ValueError("Invalid training completion, count or validation metric")
    device = metrics.get("device")
    if (not isinstance(device, str) or re.fullmatch(r"cpu|cuda:[0-9]+", device) is None
            or device != environment.get("device")
            or metrics.get("precision") not in (
                "fp32_head_fp16_cache",
                "fp32 encoders/head; pooled visual fp16 rounding matches cache")
            or config.get("precision") != "fp32_head_fp16_cache"):
        raise ValueError("Missing or mismatched validation device/precision evidence")
    if (manifest.get("architecture_id") != "hfsa_frozen_qformer_vqa_v2"
            or config.get("fusion_backend") != "blip2_qformer"
            or config.get("max_train_batches") != 0 or config.get("max_val_batches") != 0
            or training.get("validation_scope") != "full_split"
            or training.get("evaluated_val_qa") != count or labels_audit.get("val_qa") != count
            or metrics.get("dataset_sha256") != config.get("val_jsonl_sha256")
            or metrics.get("checkpoint_sha256") != manifest["checkpoint"]["sha256"]
            or abs(score - selected_score) > 1e-10):
        raise ValueError("Training and full-validation receipts do not describe one completed run")
    if digest(root / "weights/vqa_head.pt") != manifest["checkpoint"]["sha256"]:
        raise ValueError("Trained checkpoint changed")
    files = ["weights/vqa_head.pt", "config/train.json", "config/inference.json", "config/labels.json",
             "training_result.json", "best_validation.json", "environment.json", "history.csv",
             "split_isolation.json", "cache_data_binding.json", "labels_audit.json",
             "evaluation/metrics.json", "evaluation/val_predictions.jsonl", "evaluation/cache_data_binding.json"]
    if (root / "evaluation/format_predictions.json").is_file():
        files.append("evaluation/format_predictions.json")
    manifest["file_sha256"] = {relative: digest(root / relative) for relative in files}
    manifest.update(evaluation_status="complete", delivery_status="trained_full_validation",
                    metrics="evaluation/metrics.json", training_result="training_result.json",
                    validation_predictions="evaluation/val_predictions.jsonl",
                    trained_epoch=best_epoch, verified_devices=[device],
                    verified_precisions=[config["precision"]],
                    verified_device_scope="Full cached validation on recorded device; online/HTTP checked separately")
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"architecture_id": manifest["architecture_id"], "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "epochs_completed": training["epochs_completed"], "best_epoch": training["best_epoch"],
            "validation_qa": metrics["n"], "first_stage_macro_accuracy": metrics["first_stage_macro_accuracy"],
            "accuracy": metrics["accuracy"], "evaluation_status": "complete"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path("runs/vqa/qformer"))
    args = parser.parse_args()
    print(json.dumps(seal(args.bundle_dir)), flush=True)


if __name__ == "__main__":
    main()
