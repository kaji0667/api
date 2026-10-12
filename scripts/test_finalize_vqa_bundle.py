"""Synthetic receipt checks; no model/checkpoint deserialization or real data."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from finalize_vqa_bundle import digest, seal


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FinalizeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.val_hash = "a" * 64
        checkpoint = self.root / "weights/vqa_head.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic receipt checkpoint, never deserialized")
        self.checkpoint_hash = digest(checkpoint)
        self.config = {"fusion_backend": "blip2_qformer", "max_train_batches": 0,
                       "max_val_batches": 0, "val_jsonl_sha256": self.val_hash,
                       "epochs": 15, "precision": "fp32_head_fp16_cache"}
        self.training = {"validation_scope": "full_split", "evaluated_val_qa": 2,
                         "best_macro_accuracy": 0.5, "best_epoch": 2, "epochs_completed": 3}
        self.metrics = {"n": 2, "dataset_sha256": self.val_hash,
                        "checkpoint_sha256": self.checkpoint_hash,
                        "first_stage_macro_accuracy": 0.5, "accuracy": 0.5,
                        "device": "cuda:0", "precision": "fp32_head_fp16_cache"}
        entries = {"config/train.json": self.config,
                   "config/inference.json": {"fusion_backend": "blip2_qformer"},
                   "config/labels.json": {"supported": ["color"]},
                   "training_result.json": self.training,
                   "best_validation.json": {"n": 2, "first_stage_macro_accuracy": 0.5, "accuracy": 0.5},
                   "environment.json": {"device": "cuda:0"},
                   "split_isolation.json": {"train_groups": 4, "val_groups": 2, "overlap": 0},
                   "cache_data_binding.json": {"verified_datasets": []},
                   "labels_audit.json": {"train_qa": 4, "val_qa": 2},
                   "evaluation/metrics.json": self.metrics,
                   "evaluation/cache_data_binding.json": {"verified_datasets": []}}
        for relative, value in entries.items():
            write_json(self.root / relative, value)
        (self.root / "history.csv").write_text("epoch,macro_accuracy\n1,0.3\n2,0.5\n3,0.4\n")
        predictions = [{"sample_id": "val:synthetic:0", "correct": True},
                       {"sample_id": "val:synthetic:1", "correct": False}]
        (self.root / "evaluation/val_predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions))
        files = ["weights/vqa_head.pt", "config/train.json", "config/inference.json", "config/labels.json"]
        self.manifest = {"format_version": "1.0", "architecture_id": "hfsa_frozen_qformer_vqa_v2",
                         "checkpoint": {"path": "weights/vqa_head.pt", "sha256": self.checkpoint_hash},
                         "file_sha256": {relative: digest(self.root / relative) for relative in files}}
        write_json(self.root / "manifest.json", self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def update_config_receipt(self):
        write_json(self.root / "config/train.json", self.config)
        self.manifest["file_sha256"]["config/train.json"] = digest(self.root / "config/train.json")
        write_json(self.root / "manifest.json", self.manifest)

    def test_complete_full_validation_seals_and_binds_all_files(self):
        receipt = seal(self.root)
        self.assertEqual(receipt["validation_qa"], 2)
        self.assertEqual(receipt["checkpoint_sha256"], self.checkpoint_hash)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(manifest["delivery_status"], "trained_full_validation")
        self.assertEqual(manifest["evaluation_status"], "complete")
        for relative, expected in manifest["file_sha256"].items():
            self.assertEqual(digest(self.root / relative), expected)
        self.assertIn("evaluation/val_predictions.jsonl", manifest["file_sha256"])

    def test_limited_training_and_validation_are_rejected(self):
        for key in ("max_train_batches", "max_val_batches"):
            with self.subTest(key=key):
                self.config[key] = 1
                self.update_config_receipt()
                with self.assertRaises(ValueError):
                    seal(self.root)
                self.config[key] = 0
        self.update_config_receipt()
        self.training["validation_scope"] = "limited_batches"
        write_json(self.root / "training_result.json", self.training)
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_changed_checkpoint_is_rejected(self):
        (self.root / "weights/vqa_head.pt").write_bytes(b"changed checkpoint")
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_missing_dataset_hashes_cannot_match_as_none(self):
        self.config.pop("val_jsonl_sha256")
        self.metrics.pop("dataset_sha256")
        self.update_config_receipt()
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_missing_original_core_file_hash_is_rejected(self):
        self.manifest["file_sha256"].pop("config/labels.json")
        write_json(self.root / "manifest.json", self.manifest)
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_original_four_file_hash_changes_are_rejected(self):
        original = dict(self.manifest["file_sha256"])
        for relative in original:
            with self.subTest(relative=relative):
                self.manifest["file_sha256"] = {**original, relative: "0" * 64}
                write_json(self.root / "manifest.json", self.manifest)
                with self.assertRaises(ValueError):
                    seal(self.root)

    def test_illegal_dataset_and_checkpoint_hashes_are_rejected(self):
        for value in (None, "", "a" * 63, "z" * 64):
            with self.subTest(kind="dataset", value=value):
                self.config["val_jsonl_sha256"] = value
                self.metrics["dataset_sha256"] = value
                self.update_config_receipt()
                write_json(self.root / "evaluation/metrics.json", self.metrics)
                with self.assertRaises(ValueError):
                    seal(self.root)
        self.config["val_jsonl_sha256"] = self.val_hash
        self.metrics["dataset_sha256"] = self.val_hash
        self.update_config_receipt()
        for value in (None, "", "a" * 63, "z" * 64):
            with self.subTest(kind="checkpoint", value=value):
                self.manifest["checkpoint"]["sha256"] = value
                self.metrics["checkpoint_sha256"] = value
                write_json(self.root / "manifest.json", self.manifest)
                write_json(self.root / "evaluation/metrics.json", self.metrics)
                with self.assertRaises(ValueError):
                    seal(self.root)

    def test_missing_checkpoint_hashes_are_rejected(self):
        self.manifest["checkpoint"].pop("sha256")
        self.metrics.pop("checkpoint_sha256")
        write_json(self.root / "manifest.json", self.manifest)
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_illegal_best_epoch_is_rejected(self):
        for value in (None, 0, -1, 4, True):
            with self.subTest(value=value):
                self.training["best_epoch"] = value
                write_json(self.root / "training_result.json", self.training)
                with self.assertRaises(ValueError):
                    seal(self.root)

    def test_label_audit_denominator_must_agree(self):
        write_json(self.root / "labels_audit.json", {"train_qa": 4, "val_qa": 3})
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_measured_cpu_device_is_preserved(self):
        self.metrics["device"] = "cpu"
        write_json(self.root / "environment.json", {"device": "cpu"})
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        seal(self.root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(manifest["verified_devices"], ["cpu"])
        self.assertEqual(manifest["verified_precisions"], ["fp32_head_fp16_cache"])

    def test_missing_or_mismatched_measured_device_is_rejected(self):
        self.metrics.pop("device")
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        with self.assertRaises(ValueError):
            seal(self.root)
        self.metrics["device"] = "cpu"
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_missing_or_mismatched_precision_is_rejected(self):
        self.metrics.pop("precision")
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        with self.assertRaises(ValueError):
            seal(self.root)
        self.metrics["precision"] = "fp16"
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_changed_inference_configuration_is_not_resealed(self):
        write_json(self.root / "config/inference.json", {"fusion_backend": "changed"})
        with self.assertRaises(ValueError):
            seal(self.root)

    def test_nonfinite_metrics_and_mismatched_counts_are_rejected(self):
        self.metrics["first_stage_macro_accuracy"] = float("nan")
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        with self.assertRaises(ValueError):
            seal(self.root)
        self.metrics["first_stage_macro_accuracy"] = 0.5
        self.metrics["n"] = 1
        write_json(self.root / "evaluation/metrics.json", self.metrics)
        with self.assertRaises(ValueError):
            seal(self.root)


if __name__ == "__main__":
    unittest.main()
