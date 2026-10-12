"""Synthetic independent-change fixtures verify dispatch, contracts and privacy."""

import hashlib
import io
import json
import secrets
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfsa_adapter import AdapterFailure, HFSAAdapter  # noqa: E402
from model_adapter import build_model  # noqa: E402
from server import APIServer, AssetStore  # noqa: E402


@dataclass
class ChangeResult:
    supported: bool
    answer: str | None = None
    reason: str | None = None


def request(question="Synthetic temporal question", kind="short_text", **kwargs):
    constraint = {"type": kind, **kwargs.pop("constraint", {})}
    return {
        "protocol_version": "2.0", "request_id": "test:change:1",
        "item_id": "TEST_CHANGE", "answer_type": "short_text",
        "question": question, "response_constraint": constraint, **kwargs,
    }


class ChangeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.before = Path("synthetic-before.png")
        self.after = Path("synthetic-after.png")
        self.classification, self.detection, self.counting = Mock(), Mock(), Mock()
        self.refseg, self.vqa, self.change = Mock(), Mock(), Mock()
        self.change.predict.return_value = ChangeResult(True, "changed")
        self.adapter = HFSAAdapter(
            self.classification, self.detection, self.counting, self.refseg,
            vqa=self.vqa, change=self.change,
        )
        self.router_patch = patch("hfsa_adapter.route_question", return_value={"task": "change"})
        self.router = self.router_patch.start()
        self.addCleanup(self.router_patch.stop)
        # Tests must not inherit any running service's optional capture directory.
        self.capture_patch = patch.dict("os.environ", {"HFSA_API_REPLAY_DIR": ""})
        self.capture_patch.start()
        self.addCleanup(self.capture_patch.stop)

    def predict(self, value, images=None):
        self.diagnostic = io.StringIO()
        with redirect_stderr(self.diagnostic):
            return self.adapter.predict(
                value, [self.before, self.after] if images is None else images,
            )

    def assert_no_single_image_predictors_called(self):
        for predictor in (
            self.classification, self.detection, self.counting, self.refseg, self.vqa,
        ):
            predictor.predict.assert_not_called()

    def test_two_image_order_and_source_question_whitespace_are_preserved(self):
        value = request(" \tCompare Image A and Image B.\nOnly answer briefly.\r\n ")
        self.assertEqual(self.predict(value), "changed")
        self.change.predict.assert_called_once_with(
            self.before, self.after, value["question"],
            response_constraint=value["response_constraint"], choices=None,
        )
        self.router.assert_called_once_with(value)
        self.assert_no_single_image_predictors_called()

    def test_reversed_request_order_is_forwarded_without_sorting(self):
        value = request()
        self.predict(value, [self.after, self.before])
        self.change.predict.assert_called_once_with(
            self.after, self.before, value["question"],
            response_constraint=value["response_constraint"], choices=None,
        )
        self.assert_no_single_image_predictors_called()

    def test_single_choice_passes_only_declared_keys_and_returns_original_key(self):
        value = request(
            kind="single_choice", answer_type="single_choice",
            choices={"opt-left": "left", "excluded": "unsupported", "opt-right": "right"},
            constraint={"values": ["opt-right", "opt-left"]},
        )
        self.change.predict.return_value = ChangeResult(True, "opt-right")
        self.assertEqual(self.predict(value), "opt-right")
        actual = self.change.predict.call_args
        self.assertEqual(actual.args, (self.before, self.after, value["question"]))
        self.assertEqual(actual.kwargs["response_constraint"], value["response_constraint"])
        self.assertEqual(actual.kwargs["choices"], {"opt-right": "right", "opt-left": "left"})
        self.assertEqual(list(actual.kwargs["choices"]), ["opt-right", "opt-left"])
        self.assert_no_single_image_predictors_called()

    def test_region_label_is_enum_and_preserves_exact_values_and_question_form(self):
        value = request(
            kind="enum", answer_type="region_label",
            constraint={"values": ["Upper Left", "Lower Right"],
                        "question_form": "change_region", "case_insensitive": True},
        )
        self.change.predict.return_value = ChangeResult(True, "Lower Right")
        self.assertEqual(self.predict(value), "Lower Right")
        self.change.predict.assert_called_once_with(
            self.before, self.after, value["question"],
            response_constraint=value["response_constraint"], choices=None,
        )
        self.assert_no_single_image_predictors_called()

    def test_choice_text_or_excluded_key_is_not_accepted_as_answer(self):
        value = request(
            kind="single_choice", choices={"left-key": "left", "right-key": "right", "extra": "center"},
            constraint={"values": ["left-key", "right-key"]},
        )
        for answer in ("left", "extra", "LEFT-KEY"):
            with self.subTest(answer=answer):
                self.change.predict.return_value = ChangeResult(True, answer)
                with self.assertRaises(AdapterFailure) as raised:
                    self.predict(value)
                self.assertEqual(raised.exception.code, "change_answer_contract")
        self.assert_no_single_image_predictors_called()

    def test_enum_answer_must_match_an_original_allowed_value(self):
        value = request(kind="enum", constraint={"values": ["North", "South"], "case_insensitive": True})
        self.change.predict.return_value = ChangeResult(True, "south")
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(value)
        self.assertEqual(raised.exception.code, "change_answer_contract")
        self.assert_no_single_image_predictors_called()

    def test_empty_duplicate_or_missing_choice_options_fail_before_model(self):
        for values in ([], ["A", "A"], ["missing"], [""]):
            with self.subTest(values=values):
                value = request(kind="single_choice", choices={"A": "left"}, constraint={"values": values})
                with self.assertRaises(AdapterFailure) as raised:
                    self.predict(value)
                self.assertEqual(raised.exception.code, "change_invalid_options")
        self.change.predict.assert_not_called()
        self.assert_no_single_image_predictors_called()

    def test_invalid_enum_values_fail_before_model(self):
        for values in ([], ["North", "North"], [""], [1]):
            with self.subTest(values=values):
                with self.assertRaises(AdapterFailure) as raised:
                    self.predict(request(kind="enum", constraint={"values": values}))
                self.assertEqual(raised.exception.code, "change_invalid_options")
        self.change.predict.assert_not_called()
        self.assert_no_single_image_predictors_called()

    def test_short_text_minimum_length_is_checked(self):
        self.change.predict.return_value = ChangeResult(True, "new")
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request(constraint={"min_length": 4}))
        self.assertEqual(raised.exception.code, "change_answer_contract")
        self.assert_no_single_image_predictors_called()

    def test_text_length_limit_accepts_4096_and_rejects_4097_characters(self):
        self.change.predict.return_value = ChangeResult(True, "x" * 4096)
        self.assertEqual(self.predict(request()), "x" * 4096)
        self.change.predict.return_value = ChangeResult(True, "x" * 4097)
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request())
        self.assertEqual(raised.exception.code, "invalid_change_result")
        self.assert_no_single_image_predictors_called()

    def test_malformed_results_never_become_answers(self):
        results = (
            None, {}, {"supported": True, "answer": "changed", "reason": None},
            ChangeResult("yes", "changed"), ChangeResult(True),
            ChangeResult(True, 7), ChangeResult(True, ["changed"]),
            ChangeResult(True, ""), ChangeResult(True, " \t\n "),
        )
        for result in results:
            with self.subTest(result=result):
                self.change.predict.return_value = result
                with self.assertRaises(AdapterFailure) as raised:
                    self.predict(request())
                self.assertEqual(raised.exception.code, "invalid_change_result")
        self.assert_no_single_image_predictors_called()

    def test_one_or_three_images_fail_before_change_or_other_models(self):
        for images in ([self.before], [self.before, self.after, Path("synthetic-third.png")]):
            with self.subTest(image_count=len(images)):
                with self.assertRaises(AdapterFailure) as raised:
                    self.predict(request(), images)
                self.assertEqual(raised.exception.code, "unsupported_image_count")
        self.change.predict.assert_not_called()
        self.assert_no_single_image_predictors_called()

    def test_missing_change_executor_does_not_fall_back(self):
        self.adapter.change = None
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request())
        self.assertEqual(raised.exception.code, "change_not_configured")
        self.change.predict.assert_not_called()
        self.assert_no_single_image_predictors_called()

    def test_explicit_unsupported_result_uses_safe_fixed_code_without_fallback(self):
        self.change.predict.return_value = ChangeResult(False, reason="unsupported_task")
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request())
        self.assertEqual(raised.exception.code, "change_unsupported_task")
        self.assert_no_single_image_predictors_called()

    def test_unknown_or_private_failure_reason_is_rejected_and_not_logged(self):
        for reason in (None, "unsupported private question /secret/path", "unknown_reason"):
            with self.subTest(reason=reason):
                self.change.predict.return_value = ChangeResult(False, reason=reason)
                with self.assertRaises(AdapterFailure) as raised:
                    self.predict(request("private original question"))
                self.assertEqual(raised.exception.code, "invalid_change_result")
                self.assertNotIn("private original question", self.diagnostic.getvalue())
                self.assertNotIn("/secret/path", self.diagnostic.getvalue())
                self.assertNotIn("unknown_reason", self.diagnostic.getvalue())
        self.assert_no_single_image_predictors_called()

    def test_change_integer_and_bbox_contracts_fail_before_model(self):
        for kind in ("integer", "bbox"):
            with self.subTest(kind=kind):
                with self.assertRaises(AdapterFailure) as raised:
                    self.predict(request(kind=kind))
                self.assertEqual(raised.exception.code, "task_constraint_mismatch")
        self.change.predict.assert_not_called()
        self.assert_no_single_image_predictors_called()

    def test_two_image_vqa_still_cannot_fall_into_change_model(self):
        self.router.return_value = {"task": "vqa"}
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request())
        self.assertEqual(raised.exception.code, "unsupported_image_count")
        self.change.predict.assert_not_called()
        self.assert_no_single_image_predictors_called()

    def test_success_diagnostic_does_not_include_question_choices_or_answer(self):
        value = request("private original question", kind="single_choice",
                        choices={"private-key": "private choice text"}, constraint={"values": ["private-key"]})
        self.change.predict.return_value = ChangeResult(True, "private-key")
        self.assertEqual(self.predict(value), "private-key")
        output = self.diagnostic.getvalue()
        row = json.loads(output.strip())
        self.assertEqual((row["task"], row["status"]), ("change", "succeeded"))
        for private in (value["question"], "private choice text", "private-key", str(self.before), str(self.after)):
            self.assertNotIn(private, output)

    def test_unexpected_exception_diagnostic_does_not_include_error_text(self):
        self.change.predict.side_effect = RuntimeError("private question /secret/path secret-key")
        with self.assertRaises(RuntimeError):
            self.predict(request("private original question"))
        output = self.diagnostic.getvalue()
        row = json.loads(output.strip())
        self.assertEqual((row["task"], row["status"], row["reason"]),
                         ("change", "failed", "unexpected_exception"))
        for private in ("private question", "/secret/path", "secret-key", "private original question"):
            self.assertNotIn(private, output)
        self.assert_no_single_image_predictors_called()


class ChangeHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name)
        assets = self.package / "assets"
        assets.mkdir()
        self.entries = []
        for character, payload in (("a", b"synthetic before"), ("b", b"synthetic after")):
            asset_id = character * 32
            (assets / asset_id).write_bytes(payload)
            self.entries.append({"asset_id": asset_id, "sha256": hashlib.sha256(payload).hexdigest(),
                                 "size": len(payload), "mime_type": "image/png"})
        (self.package / "manifest.json").write_text(json.dumps({
            "format": "rsu-sealed-images-v1", "dataset_id": "synthetic-change-v1", "assets": self.entries,
        }), encoding="utf-8")
        self.classification, self.detection, self.counting = Mock(), Mock(), Mock()
        self.vqa, self.change = Mock(), Mock()
        self.change.predict.return_value = ChangeResult(True, "changed")
        adapter = HFSAAdapter(self.classification, self.detection, self.counting, None,
                              vqa=self.vqa, change=self.change)
        self.adapter = adapter
        with patch.dict("os.environ", {"MODEL_MODE": "real"}), \
                patch.object(HFSAAdapter, "from_local", return_value=adapter):
            model = build_model()
        self.api_key = secrets.token_urlsafe(32)
        self.server = APIServer(("127.0.0.1", 0), AssetStore(self.package), model, self.api_key)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.router_patch = patch("hfsa_adapter.route_question", return_value={"task": "change"})
        self.router_patch.start()
        self.addCleanup(self.router_patch.stop)
        self.capture_patch = patch.dict("os.environ", {"HFSA_API_REPLAY_DIR": ""})
        self.capture_patch.start()
        self.addCleanup(self.capture_patch.stop)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request_data(self):
        return request(images=[{key: entry[key] for key in ("asset_id", "sha256", "mime_type")}
                               for entry in self.entries])

    def call(self, value):
        call = urllib.request.Request(
            self.origin + "/v1/predict", data=json.dumps(value).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with redirect_stderr(io.StringIO()):
            try:
                with opener.open(call, timeout=3) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

    def test_real_model_http_returns_all_three_string_contracts(self):
        cases = (
            ("short_text", "short_text", {"type": "short_text"}, None, "changed"),
            ("single_choice", "single_choice", {"type": "single_choice", "values": ["right-key"]},
             {"right-key": "right", "excluded": "left"}, "right-key"),
            ("enum", "region_label", {"type": "enum", "values": ["South"],
                                      "question_form": "change_region"}, None, "South"),
        )
        for kind, answer_type, constraint, choices, expected in cases:
            with self.subTest(kind=kind):
                value = self.request_data()
                value.update(answer_type=answer_type, response_constraint=constraint)
                if choices is not None:
                    value["choices"] = choices
                self.change.predict.return_value = ChangeResult(True, expected)
                status, result = self.call(value)
                self.assertEqual(status, 200)
                self.assertEqual(result, {"protocol_version": "2.0", "request_id": value["request_id"],
                                          "item_id": value["item_id"], "answer": expected})
        self.assertEqual(self.change.predict.call_count, 3)
        for predictor in (self.classification, self.detection, self.counting, self.vqa):
            predictor.predict.assert_not_called()

    def test_server_verified_paths_keep_request_array_order(self):
        value = self.request_data()
        value["images"].reverse()
        status, _ = self.call(value)
        self.assertEqual(status, 200)
        actual = self.change.predict.call_args.args
        self.assertEqual(actual[:2], (self.package / "assets" / ("b" * 32),
                                     self.package / "assets" / ("a" * 32)))

    def test_bad_second_asset_is_rejected_before_change_inference(self):
        value = self.request_data()
        value["images"][1]["sha256"] = "0" * 64
        self.assertEqual(self.call(value), (422, {"error": "asset_metadata_mismatch"}))
        self.change.predict.assert_not_called()

    def test_invalid_change_answer_is_generic_http_failure_without_details(self):
        value = self.request_data()
        value["response_constraint"] = {"type": "enum", "values": ["North", "South"]}
        self.change.predict.return_value = ChangeResult(True, "private disallowed answer")
        self.assertEqual(self.call(value), (500, {"error": "inference_failed"}))

    def binary_predictor(self, region_pixels):
        from tasks.change.answering import EvidenceAnswerer
        from tasks.change.inference import ChangePredictor

        predictor = ChangePredictor.__new__(ChangePredictor)
        summary = {"mask_scope": "generic_binary_change", "mask_grid": {"binary": {
            "region_pixels": region_pixels, "changed_pixels": sum(region_pixels.values())}}}
        predictor.binary_predictor = Mock()
        predictor.binary_predictor.evidence.return_value = SimpleNamespace(summary=lambda: summary)
        predictor.evidence = Mock(side_effect=AssertionError("Generic regions must not execute MCI"))
        predictor.answerer = EvidenceAnswerer()
        self.adapter.change = predictor
        return predictor

    def test_generic_binary_regions_follow_real_answer_dispatch_and_original_contracts(self):
        names = ["top_left", "top_center", "top_right", "middle_left", "center",
                 "middle_right", "bottom_left", "bottom_center", "bottom_right"]
        counts = {name: 0 for name in names}
        counts.update(bottom_center=20, upper_left=0, upper_right=0, lower_left=20, lower_right=0)
        predictor = self.binary_predictor(counts)
        value = self.request_data()
        value.update(question="Which region shows the most significant change?", answer_type="region_label",
                     response_constraint={"type": "enum", "question_form": "change_region",
                                          "values": ["top_left", "top", "top_right", "left", "center",
                                                     "right", "bottom_left", "bottom", "bottom_right"]})
        status, result = self.call(value)
        self.assertEqual((status, result["answer"]), (200, "bottom"))
        predictor.binary_predictor.evidence.assert_called_once_with(
            self.package / "assets" / ("a" * 32), self.package / "assets" / ("b" * 32))
        value.update(answer_type="single_choice", choices={"b": "Lower Right", "a": "Upper Left",
                      "d": "Lower Left", "c": "Upper Right", "excluded": "Unknown"},
                     response_constraint={"type": "single_choice", "question_form": "change_region",
                                          "values": ["b", "a", "d", "c"]})
        value["images"].reverse()
        status, result = self.call(value)
        self.assertEqual((status, result["answer"]), (200, "d"))
        self.assertEqual(predictor.binary_predictor.evidence.call_args.args,
                         (self.package / "assets" / ("b" * 32), self.package / "assets" / ("a" * 32)))
        predictor.evidence.assert_not_called()

    def test_generic_binary_region_zero_and_tie_fail_without_guessed_answer(self):
        names = ["upper_left", "upper_right", "lower_left", "lower_right"]
        value = self.request_data()
        value.update(question="Which quadrant has changed the most?", answer_type="region_label",
                     response_constraint={"type": "enum", "question_form": "change_region", "values": names})
        for amount in (0, 10):
            with self.subTest(amount=amount):
                predictor = self.binary_predictor(dict.fromkeys(names, amount))
                self.assertEqual(self.call(value), (500, {"error": "inference_failed"}))
                predictor.evidence.assert_not_called()


if __name__ == "__main__":
    unittest.main()
