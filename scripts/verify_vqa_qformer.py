"""Verify a trained Q-Former checkpoint online against its immutable feature cache."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tasks.vqa.cache import FeatureCache
from tasks.vqa.config import QFORMER_ARCHITECTURE_ID, SUPPORTED, sha256, write_json
from tasks.vqa.data import read_rows
from tasks.vqa.inference import VQAPredictor, decode_prediction
from tasks.vqa.model import unpack_visual


def assert_same_result(actual, cached):
    for key in ("status", "answer", "answer_text", "question_type", "reason", "evidence"):
        if actual.get(key) != cached.get(key):
            raise AssertionError(f"Online/cache result differs: {key}")
    if actual.get("confidence") is not None:
        if abs(actual["confidence"] - cached["confidence"]) > 5e-4:
            raise AssertionError("Online/cache confidence exceeds numerical tolerance")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--jsonl", default="../vqa_work_20261011/data/smoke_val.jsonl")
    parser.add_argument("--cache-dir", default="../vqa_work_20261011/cache/full")
    parser.add_argument("--visual-backbone", default="../HFSA_models/pretrain_model/yolov12m.pt")
    parser.add_argument("--text-encoder", default="../HFSA_models/openclip/ViT-L-14.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/vqa/qformer/reports/online_verification.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    assets = dict(visual_backbone=args.visual_backbone, text_encoder=args.text_encoder)
    predictor = VQAPredictor(args.bundle_dir, args.device, assets)
    if predictor.manifest["architecture_id"] != QFORMER_ARCHITECTURE_ID:
        raise ValueError("This verification requires a trained Q-Former v2 checkpoint")
    if predictor.visual.training or predictor.text.encoder.training:
        raise AssertionError("Frozen encoders must stay in eval mode")
    if any(p.requires_grad for p in predictor.visual.parameters()) or any(p.requires_grad for p in predictor.text.parameters()):
        raise AssertionError("Shared encoder parameters are not frozen")
    head_identity = id(predictor.head)
    dataset = read_rows(args.jsonl)
    rows = []
    for kind in SUPPORTED:
        match = next((row for row in dataset if row["question_type"] == kind), None)
        if match is not None:
            rows.append(match)
    if not rows:
        raise ValueError("No supported source question types available")
    cache = FeatureCache(args.cache_dir, rows, {**predictor.config, **assets}, load_ram=False)
    binding = cache.verify_datasets([args.jsonl])
    records = []
    formats_checked = []
    for index, row in enumerate(rows):
        flat, question = cache.batch([index], predictor.device)
        type_logits, answer_logits = predictor.head(
            unpack_visual(flat, predictor.config), question,
            predictor.config.get("train_ablation"),
        )
        logits = {key: value[0] for key, value in answer_logits.items()}
        image_size = (row["width"], row["height"])
        cached = decode_prediction(type_logits[0], logits, predictor.labels,
            row["question"], image_size=image_size)
        actual = predictor.predict(row["image_path"], row["question"])
        assert_same_result(actual, cached)
        records.append(dict(sample_id=row["sample_id"], source_type=row["question_type"],
                            mode="short_text", result=actual))
        if actual["status"] == "ok" and not formats_checked:
            names = predictor.labels["answers"][actual["question_type"]][:2]
            choices = {f"option-{i + 1}": name for i, name in enumerate(names)}
            modes = [("single_choice", dict(choices=choices)),
                     ("enum", dict(allowed_answers=list(names)))]
            for mode, options in modes:
                cached_format = decode_prediction(type_logits[0], logits, predictor.labels,
                    row["question"], image_size=image_size, **options)
                actual_format = predictor.predict(row["image_path"], row["question"], **options)
                assert_same_result(actual_format, cached_format)
                records.append(dict(sample_id=row["sample_id"], mode=mode, result=actual_format))
                formats_checked.append(mode)
    if id(predictor.head) != head_identity:
        raise AssertionError("Predictor replaced its model between requests")
    report = dict(
        architecture_id=predictor.manifest["architecture_id"], device=str(predictor.device),
        checkpoint_sha256=sha256(Path(args.bundle_dir) / "weights/vqa_head.pt"),
        new_learned_parameters=sum(p.numel() for p in predictor.head.parameters()),
        frozen_backbone_load_report=predictor.visual.load_report,
        online_cache_pairs=len(records), formats_checked=formats_checked,
        encoders_frozen_and_eval=True, persistent_predictor=True,
        cache_data_binding=binding, records=records,
        scope="Integration verification only; no accuracy or improvement claim",
    )
    write_json(args.out, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"records", "cache_data_binding"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
