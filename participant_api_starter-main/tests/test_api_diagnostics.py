"""Private diagnostic persistence, actual executor tracking and fail-open checks."""

import io
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfsa_adapter import AdapterFailure, HFSAAdapter, _append_prediction_diagnostic


class PersistentDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.log = Path(self.directory.name) / "private" / "predictions.jsonl"
        self.environment = patch.dict(os.environ, {
            "HFSA_API_DIAGNOSTIC_FILE": str(self.log), "HFSA_API_REPLAY_DIR": "",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.router_patch = patch("hfsa_adapter.route_question", return_value={"task": "counting"})
        self.route = self.router_patch.start()
        self.addCleanup(self.router_patch.stop)
        self.counting = Mock()
        self.counting.predict.return_value = SimpleNamespace(count=3)
        self.classification = Mock(class_names=["bridge"])
        self.classification.predict.return_value = {
            "predictions": [{"class_name": "bridge", "probability": 1.0}],
        }
        self.detection = Mock()
        self.detection.predict.return_value = SimpleNamespace(detection_count=2)
        self.vqa = Mock()
        self.vqa.predict.return_value = {
            "status": "ok", "answer": "private-model-answer", "question_type": "category",
        }
        self.adapter = HFSAAdapter(self.classification, self.detection, self.counting,
                                   None, vqa=self.vqa)
        self.output = io.StringIO()

    def predict(self, task="counting", kind="integer", image_count=1, **extra):
        self.route.return_value = {"task": task}
        request = {
            "question": "private-question-text", "response_constraint": {"type": kind},
            "item_id": "SYNTHETIC_ITEM", "request_id": "synthetic:run:1",
            "choices": {"private-choice": "private-choice-text"},
            "headers": {"Authorization": "private-bearer-token"},
            "api_key": "private-api-key", "answer": "private-gold-answer", **extra,
        }
        images = [Path("/private/image-path")] * image_count
        with redirect_stderr(self.output):
            return self.adapter.predict(request, images)

    def rows(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def test_append_records_match_stderr_and_exclude_content(self):
        self.assertEqual(self.predict(), "3")
        self.assertEqual(self.predict("vqa", "short_text"), "private-model-answer")
        rows = self.rows()
        self.assertEqual(rows, [json.loads(line) for line in self.output.getvalue().splitlines()])
        self.assertEqual([row["executor"] for row in rows], ["counting", "vqa"])
        for row in rows:
            self.assertEqual(row["status"], "succeeded")
            self.assertIsNone(row["reason"])
            self.assertGreaterEqual(row["elapsed_ms"], 0)
            self.assertGreater(row["process_id"], 0)
            self.assertRegex(row["timestamp_utc"], r"\d{4}-\d{2}-\d{2}T.*\+00:00$")
        self.assertNotIn("private-", self.log.read_text(encoding="utf-8"))
        self.assertNotIn("answer", rows[0])
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.log.stat().st_mode), 0o600)

    def test_classification_and_presence_record_actual_predictors(self):
        self.assertEqual(self.predict("classification", "short_text"), "bridge")
        self.predict("presence", "short_text")
        self.assertEqual([row["executor"] for row in self.rows()], ["classification", "detection"])

    def test_known_rejection_records_executor_and_fixed_code(self):
        self.vqa.predict.return_value = {"status": "unsupported", "reason": "unsupported_task"}
        with self.assertRaises(AdapterFailure) as raised:
            self.predict("vqa", "short_text")
        self.assertEqual(raised.exception.code, "vqa_unsupported_task")
        row = self.rows()[0]
        self.assertEqual((row["task"], row["executor"], row["status"], row["reason"]),
                         ("vqa", "vqa", "failed", "vqa_unsupported_task"))

    def test_two_image_rejection_precedes_routing_and_model_execution(self):
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(image_count=2)
        self.assertEqual(raised.exception.code, "unsupported_image_count")
        row = self.rows()[0]
        self.assertIsNone(row["task"])
        self.assertIsNone(row["executor"])
        self.assertEqual(row["reason"], "unsupported_image_count")
        self.route.assert_not_called()
        for predictor in (self.classification, self.detection, self.counting, self.vqa):
            predictor.predict.assert_not_called()

    def test_unknown_code_and_exception_class_cannot_leak_arbitrary_text(self):
        self.counting.predict.side_effect = AdapterFailure("private-code-token", "private-message")
        with self.assertRaises(AdapterFailure):
            self.predict()
        unsafe_class = type("private_class_token", (RuntimeError,), {})
        self.counting.predict.side_effect = unsafe_class("private-exception-key")
        with self.assertRaises(RuntimeError):
            self.predict()
        rows = self.rows()
        self.assertEqual([row["reason"] for row in rows], ["unexpected_exception"] * 2)
        self.assertEqual(rows[1]["error_type"], "Exception")
        self.assertNotIn("private", self.log.read_text(encoding="utf-8"))

    def test_invalid_file_destination_does_not_change_answer_or_rejection(self):
        self.log.parent.mkdir()
        self.log.mkdir()
        self.assertEqual(self.predict(), "3")
        self.counting.predict.side_effect = AdapterFailure("count_option_unmatched", "private-message")
        with self.assertRaises(AdapterFailure) as raised:
            self.predict()
        self.assertEqual(raised.exception.code, "count_option_unmatched")
        self.assertEqual(len(self.output.getvalue().splitlines()), 2)
        self.assertNotIn(str(self.log), self.output.getvalue())

    def test_unavailable_stderr_does_not_prevent_file_append(self):
        with patch("hfsa_adapter.print", side_effect=OSError("private-stderr-message")):
            self.assertEqual(self.predict(), "3")
        self.assertEqual(self.rows()[0]["status"], "succeeded")

    def test_disabled_persistence_keeps_only_existing_stderr_diagnostic(self):
        with patch.dict(os.environ, {"HFSA_API_DIAGNOSTIC_FILE": ""}):
            self.assertEqual(self.predict(), "3")
        self.assertFalse(self.log.exists())
        self.assertEqual(json.loads(self.output.getvalue())["status"], "succeeded")

    @unittest.skipUnless(os.name == "posix", "POSIX no-follow path check")
    def test_symlink_destination_is_not_followed(self):
        self.log.parent.mkdir()
        target = self.log.parent / "existing-private-file"
        target.write_text("unchanged", encoding="utf-8")
        self.log.symlink_to(target)
        self.assertEqual(self.predict(), "3")
        self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")

    def test_concurrent_rows_remain_separate_json_objects(self):
        threads = [threading.Thread(target=_append_prediction_diagnostic,
                                    args=(json.dumps({"synthetic": number}),))
                   for number in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(row["synthetic"] for row in self.rows()), list(range(8)))

    def test_evaluation_enclosure_diagnostics_keep_fixed_values_without_bbox_or_source(self):
        from tasks.detection.spatial_reference_grounding import ReferenceEnclosureResult

        executor = Mock()
        executor.predict.return_value = ReferenceEnclosureResult((1, 2, 20, 30), (40, 80),
            diagnostics=dict(grounding_policy="reference_enclosure", grounding_parser="qwen",
                             grounding_parse_status="parsed", grounding_outcome="enclosure_selected",
                             grounding_reason="grounding_reference_enclosed", grounding_reference_count=4,
                             grounding_located_reference_count=4, grounding_roi_count=0,
                             grounding_detector_call_count=4, bbox=[1, 2, 20, 30],
                             source="private-source", confidence=.99))
        self.adapter = HFSAAdapter(None, self.detection, None, None,
                                   grounding_policy="reference_enclosure", grounding_executor=executor)
        self.predict("visual_grounding", "bbox", image_width=80, image_height=40)
        safe = self.rows()[0]["detection_inference"]
        self.assertEqual(safe["grounding_policy"], "reference_enclosure")
        self.assertEqual(safe["grounding_outcome"], "enclosure_selected")
        self.assertEqual(safe["grounding_reason"], "grounding_reference_enclosed")
        self.assertEqual(safe["grounding_roi_count"], 0)
        self.assertNotIn("bbox", safe)
        self.assertNotIn("confidence", safe)
        self.assertNotIn("private", self.log.read_text(encoding="utf-8"))

    def test_spatial_parser_diagnostics_keep_fixed_reason_and_counts_only(self):
        import numpy as np
        self.detection.predict.return_value = SimpleNamespace(
            detection_count=1, original_size=(40, 80), boxes=np.array([[1, 2, 20, 30]]),
            scores=np.array([.5]), grounding_index=0,
            diagnostics=dict(grounding_policy="spatial_reference", grounding_parser="qwen",
                             grounding_parse_status="unsupported", grounding_outcome="native_fallback",
                             grounding_reason="grounding_parser_source_mismatch",
                             grounding_reference_count=4, grounding_container_count=1,
                             grounding_located_container_count=0, source="private-source-span",
                             parsed_plan="private-parser-output"),
        )
        self.predict("visual_grounding", "bbox", image_width=80, image_height=40)
        safe = self.rows()[0]["detection_inference"]
        self.assertEqual(safe["grounding_parser"], "qwen")
        self.assertEqual(safe["grounding_reason"], "grounding_parser_source_mismatch")
        self.assertEqual(safe["grounding_container_count"], 1)
        self.assertNotIn("private", self.log.read_text(encoding="utf-8"))

    def test_reference_attempts_keep_numbers_and_fixed_statuses_without_descriptions_or_boxes(self):
        from tasks.detection.spatial_reference_grounding import ReferenceEnclosureResult

        executor = Mock()
        executor.predict.return_value = ReferenceEnclosureResult((1, 2, 20, 30), (40, 80), diagnostics={
            "grounding_reference_results": [
                dict(reference_index=1, status="selected", candidate_count=2, max_confidence=.876543219,
                     selected_index=1, query="private-query", description="private-description",
                     selected_bbox_original_xyxy=[1, 2, 20, 30]),
                dict(reference_index=2, status="no_candidates", candidate_count=0, max_confidence=.01,
                     selected_index=None),
                dict(reference_index=3, status="error", failure_stage="matching", error="private-error",
                     candidate_count=True, selected_index=-1, max_confidence=float("nan")),
                dict(reference_index=True, status="selected"),
                dict(reference_index=4, status="private-status"),
                "private-malformed-attempt",
            ],
        })
        self.adapter = HFSAAdapter(None, self.detection, None, None,
                                   grounding_policy="reference_enclosure", grounding_executor=executor)
        self.predict("visual_grounding", "bbox", image_width=80, image_height=40)
        attempts = self.rows()[0]["detection_inference"]["grounding_reference_results"]
        self.assertEqual(attempts, [
            dict(reference_index=1, status="selected", candidate_count=2, max_confidence=.87654322, selected_index=1),
            dict(reference_index=2, status="no_candidates", candidate_count=0, max_confidence=.01),
            dict(reference_index=3, status="error", failure_stage="matching"),
        ])
        self.assertNotIn("private", self.log.read_text(encoding="utf-8"))
        self.assertNotIn("selected_bbox_original_xyxy", str(attempts))


if __name__ == "__main__":
    unittest.main()
