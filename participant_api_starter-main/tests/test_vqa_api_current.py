"""Boundary checks for the single current Q-Former API verifier."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR / "scripts"))
sys.path.insert(0, str(CODE_DIR / "participant_api_starter-main"))
from verify_vqa_api import (
    FixedVQAModel, disable_verification_capture, example_path, expected_http,
    protocol_request, verify_adapter,
)
from hfsa_adapter import AdapterFailure, HFSAAdapter, validate_api_vqa_bundle


def full_bundle_fixture(root):
    receipts = {
        "training_result.json": {"validation_scope": "full_split", "evaluated_val_qa": 37392,
                                 "epochs_completed": 15, "best_epoch": 13,
                                 "best_macro_accuracy": 0.5},
        "config/train.json": {"max_train_batches": 0, "max_val_batches": 0,
                              "val_jsonl_sha256": "b" * 64, "fusion_backend": "blip2_qformer"},
        "evaluation/metrics.json": {"n": 37392, "checkpoint_sha256": "a" * 64,
                                    "dataset_sha256": "b" * 64, "first_stage_macro_accuracy": 0.5},
    }
    manifest = {"architecture_id": "hfsa_frozen_qformer_vqa_v2", "fusion_backend": "blip2_qformer",
                "evaluation_status": "complete", "checkpoint": {"sha256": "a" * 64}, "file_sha256": {}}
    write_fixture(root, manifest, receipts)
    return manifest, receipts


def write_fixture(root, manifest, receipts):
    for relative, value in receipts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value).encode("utf-8")
        path.write_bytes(payload)
        manifest["file_sha256"][relative] = hashlib.sha256(payload).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class CurrentVQAAPIVerifierTests(unittest.TestCase):
    def test_public_example_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            root.mkdir()
            outside = Path(directory) / "outside.jpg"
            outside.write_bytes(b"public synthetic bytes")
            with self.assertRaises(ValueError):
                example_path(root, "../outside.jpg")

    def test_choice_keys_and_complete_question_survive_http_contract(self):
        row = {"mode": "single_choice", "question": "  What color is the northern vehicle?\n",
               "choices": {"Option-X": "Red", "key:2": "Black"}}
        request = protocol_request(row, {"asset_id": "public"}, 7)
        self.assertEqual(request["question"], row["question"])
        self.assertEqual(request["response_constraint"], {"type": "single_choice", "values": ["Option-X", "key:2"]})
        self.assertEqual(expected_http(request, {"status": "ok", "answer": "key:2"})[1]["answer"], "key:2")
        self.assertEqual(expected_http(request, {"status": "unsupported", "answer": None}),
                         (500, {"error": "inference_failed"}))

    def test_fixed_task_calls_real_adapter_without_router_or_text_changes(self):
        seen = []

        def predict(image, question, **options):
            seen.append((image, question, options))
            return {"status": "ok", "answer": "B", "answer_text": "black", "question_type": "color"}

        adapter = HFSAAdapter(None, None, None, None, vqa=SimpleNamespace(predict=predict))
        row = {"mode": "single_choice", "question": "What is the color of the vehicle?",
               "choices": {"A": "red", "B": "black"}}
        request = protocol_request(row, {"asset_id": "public"}, 0)
        with patch("hfsa_adapter.route_question", side_effect=AssertionError("Fixed-task group must not query the router")):
            self.assertEqual(FixedVQAModel(adapter).predict(request, [Path("public.jpg")]), "B")
        self.assertEqual(seen, [(Path("public.jpg"), row["question"], {"choices": row["choices"]})])
        with self.assertRaises(AdapterFailure):
            FixedVQAModel(adapter).predict(request, [Path("a.jpg"), Path("b.jpg")])
        self.assertEqual(len(seen), 1)

    def test_replay_capture_is_disabled_and_restored(self):
        with patch.dict(os.environ, {"HFSA_API_REPLAY_DIR": "public-test-location", "HFSA_TEST_DUMMY_SETTING": "retained"}):
            with disable_verification_capture():
                self.assertEqual(os.environ["HFSA_API_REPLAY_DIR"], "")
                self.assertEqual(os.environ["HFSA_TEST_DUMMY_SETTING"], "retained")
            self.assertEqual(os.environ["HFSA_API_REPLAY_DIR"], "public-test-location")

    def test_old_predictor_fails_before_http_or_qwen(self):
        old = SimpleNamespace(manifest={"architecture_id": "hfsa_frozen_spatial_vqa_v1"})
        with patch("ollama_router.route_question", side_effect=AssertionError("Old model must fail first")):
            with self.assertRaises(ValueError):
                verify_adapter(SimpleNamespace(vqa=old), Path("absent-bundle"), Path("absent-examples"))

    def test_complete_full_training_receipt_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_bundle_fixture(root)
            receipt = validate_api_vqa_bundle(root)
            self.assertEqual(receipt["validation_qa"], 37392)
            self.assertEqual(receipt["epochs_completed"], 15)
            self.assertEqual(receipt["validation_scope"], "full_split")

    def test_old_pending_and_wrong_fusion_bundles_are_rejected(self):
        for field, value in (("architecture_id", "hfsa_frozen_spatial_vqa_v1"),
                             ("evaluation_status", "pending"), ("fusion_backend", "spatial_attention")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, receipts = full_bundle_fixture(root)
                manifest[field] = value
                write_fixture(root, manifest, receipts)
                with self.assertRaises(ValueError):
                    validate_api_vqa_bundle(root)

    def test_limited_training_or_validation_cannot_be_served(self):
        cases = (("training_result.json", "validation_scope", "limited_batches"),
                 ("config/train.json", "max_train_batches", 5),
                 ("config/train.json", "max_val_batches", 2),
                 ("training_result.json", "epochs_completed", 0))
        for relative, field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, receipts = full_bundle_fixture(root)
                receipts[relative][field] = value
                write_fixture(root, manifest, receipts)
                with self.assertRaises(ValueError):
                    validate_api_vqa_bundle(root)

    def test_missing_or_tampered_receipt_cannot_be_served(self):
        for mutation in ("missing_file", "missing_hash", "changed_bytes"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, receipts = full_bundle_fixture(root)
                path = root / "training_result.json"
                if mutation == "missing_file":
                    path.unlink()
                elif mutation == "missing_hash":
                    del manifest["file_sha256"]["training_result.json"]
                    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                else:
                    path.write_text("{}", encoding="utf-8")
                with self.assertRaises((ValueError, OSError)):
                    validate_api_vqa_bundle(root)

    def test_wrong_evaluation_binding_or_denominator_is_rejected(self):
        for field, value in (("n", 16), ("checkpoint_sha256", "c" * 64), ("dataset_sha256", "c" * 64)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, receipts = full_bundle_fixture(root)
                receipts["evaluation/metrics.json"][field] = value
                write_fixture(root, manifest, receipts)
                with self.assertRaises(ValueError):
                    validate_api_vqa_bundle(root)

    def test_missing_or_malformed_source_hashes_cannot_match_each_other(self):
        for binding in ("checkpoint", "dataset"):
            for invalid in (None, "", "a" * 63, "z" * 64):
                with self.subTest(binding=binding, invalid=invalid), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest, receipts = full_bundle_fixture(root)
                    source = manifest["checkpoint"] if binding == "checkpoint" else receipts["config/train.json"]
                    source_field = "sha256" if binding == "checkpoint" else "val_jsonl_sha256"
                    metrics_field = f"{binding}_sha256"
                    if invalid is None:
                        del source[source_field]
                        del receipts["evaluation/metrics.json"][metrics_field]
                    else:
                        source[source_field] = invalid
                        receipts["evaluation/metrics.json"][metrics_field] = invalid
                    write_fixture(root, manifest, receipts)
                    with self.assertRaises(ValueError):
                        validate_api_vqa_bundle(root)

    def test_invalid_best_epoch_is_rejected_with_value_error(self):
        for best_epoch in (13.0, "13", None, True, 0, 16):
            with self.subTest(best_epoch=best_epoch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, receipts = full_bundle_fixture(root)
                receipts["training_result.json"]["best_epoch"] = best_epoch
                write_fixture(root, manifest, receipts)
                with self.assertRaises(ValueError):
                    validate_api_vqa_bundle(root)

    def test_nonfinite_or_mismatched_macro_accuracy_is_rejected(self):
        cases = ((float("nan"), float("nan")), (float("inf"), float("inf")),
                 (None, 0.5), (0.5, None), (True, True), (-0.1, -0.1),
                 (1.1, 1.1), (0.5, 0.6))
        for measured, selected in cases:
            with self.subTest(measured=measured, selected=selected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, receipts = full_bundle_fixture(root)
                receipts["evaluation/metrics.json"]["first_stage_macro_accuracy"] = measured
                receipts["training_result.json"]["best_macro_accuracy"] = selected
                write_fixture(root, manifest, receipts)
                with self.assertRaises(ValueError):
                    validate_api_vqa_bundle(root)

    def test_training_config_must_use_qformer(self):
        for fusion in (None, "spatial_attention"):
            with self.subTest(fusion=fusion), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, receipts = full_bundle_fixture(root)
                if fusion is None:
                    del receipts["config/train.json"]["fusion_backend"]
                else:
                    receipts["config/train.json"]["fusion_backend"] = fusion
                write_fixture(root, manifest, receipts)
                with self.assertRaises(ValueError):
                    validate_api_vqa_bundle(root)


if __name__ == "__main__":
    unittest.main()
