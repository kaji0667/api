"""Synthetic checks for early raw parsing and guarded, pool-bound bbox selection."""

from dataclasses import dataclass, replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tasks.detection.bbox_grounding import BBoxGroundingExecutor
from tasks.detection.crop_identity import CropIdentityError
from tasks.detection.reference_plan import parse_reference_plan


@dataclass(frozen=True)
class NativeResult:
    boxes: np.ndarray
    scores: np.ndarray
    original_size: tuple[int, int]
    grounding_index: int | None
    model_prompt: str
    diagnostics: dict | None = None
    phrase_scores: np.ndarray | None = None
    baseline_adjusted_scores: np.ndarray | None = None


class BBoxGroundingTests(unittest.TestCase):
    def setUp(self):
        self.description = "The lower-left lawn\nA road is to the right of it"
        self.question = "Description:\n" + self.description + "\nReturn a bbox."
        self.result = NativeResult(
            boxes=np.array([[5., 5., 15., 15.], [10., 50., 30., 70.]], dtype=np.float32),
            scores=np.array([.95, .30], dtype=np.float32), original_size=(80, 80),
            grounding_index=0, model_prompt=self.description.replace("\n", " "),
            diagnostics={"nms_candidate_count": 2},
            phrase_scores=np.array([1.04, 1.], dtype=np.float32),
            baseline_adjusted_scores=np.array([1.04, 1.], dtype=np.float32),
        )
        self.native = Mock(candidate_confidence=.25, spatial_weight=.15)
        self.native.predict.return_value = self.result
        self.scorer = Mock(preprocess_configuration={}, batch_size=8)
        self.contrast = [0., 1.]
        self.scorer.score.side_effect = self.score
        self.extract = Mock(side_effect=lambda question: self.description)
        self.image = object()
        self.bgr = np.zeros((80, 80, 3), dtype=np.uint8)
        self.convert = Mock(return_value=self.bgr)
        self.executor = BBoxGroundingExecutor(
            self.native, self.scorer, description_extractor=self.extract, image_converter=self.convert,
        )

    def score(self, image, boxes, target, references):
        return {
            "records": [{"index": index, "box_xyxy": box.tolist(), "status": "succeeded",
                         "target_score": value, "contrast_score": value}
                        for index, (box, value) in enumerate(zip(boxes, self.contrast))],
            "encoded_crop_count": len(boxes), "image_encode_batch_count": 1,
        }

    def test_raw_plan_is_parsed_before_native_and_original_question_is_unchanged(self):
        events = []
        self.extract.side_effect = lambda question: events.append(("extract", question)) or self.description
        self.native.predict.side_effect = lambda *a, **kw: events.append(("native", a[1])) or self.result
        def parse(description):
            events.append(("parse", description))
            return parse_reference_plan(description)
        with patch("tasks.detection.bbox_grounding.parse_reference_plan", side_effect=parse):
            result = self.executor.predict(self.image, self.question)
        self.assertEqual(events[:3], [("extract", self.question), ("parse", self.description),
                                      ("native", self.question)])
        self.native.predict.assert_called_once_with(self.image, self.question, return_phrase_scores=True)
        self.assertEqual(result.grounding_index, 1)
        self.assertEqual(result.diagnostics["grounding_reference_count"], 1)

    def test_multiline_english_and_chinese_clauses_reach_existing_scorer_raw(self):
        for description in (
            "The lower-left lawn\nA road is to the right of it",
            "The lawn\nIt is next to a road\nA building is above it",
            "左下角的梯形草坪\n右侧有道路\n上方有仓库",
        ):
            with self.subTest(description=description):
                self.description = description
                plan = parse_reference_plan(description)
                self.assertIsNotNone(plan)
                self.scorer.score.reset_mock()
                result = self.executor.predict(self.image, self.question)
                args = self.scorer.score.call_args.args
                self.assertIs(args[0], self.bgr)
                self.assertEqual(args[2:], (plan.target_description, [ref.description for ref in plan.references]))
                self.assertEqual(result.diagnostics["grounding_parse_status"], "parsed")

    def test_candidates_confidence_order_geometry_prompt_and_scores_are_preserved(self):
        result = self.executor.predict(self.image, self.question)
        self.assertIs(result.boxes, self.result.boxes)
        self.assertIs(result.scores, self.result.scores)
        self.assertIs(result.phrase_scores, self.result.phrase_scores)
        self.assertIs(result.baseline_adjusted_scores, self.result.baseline_adjusted_scores)
        self.assertEqual(result.original_size, self.result.original_size)
        self.assertEqual(result.model_prompt, self.result.model_prompt)
        self.assertEqual(self.result.grounding_index, 0)
        self.assertEqual(result.grounding_index, 1)
        np.testing.assert_array_equal(self.scorer.score.call_args.args[1], self.result.boxes)

    def test_unsupported_description_keeps_native_call_and_skips_clip_and_image_conversion(self):
        self.description = "The lawn between two roads"
        result = self.executor.predict(self.image, self.question)
        self.native.predict.assert_called_once_with(self.image, self.question)
        self.scorer.score.assert_not_called()
        self.convert.assert_not_called()
        self.assertEqual(result.grounding_index, 0)
        self.assertEqual(result.diagnostics["grounding_reason"], "grounding_unsupported_plan")

    def test_empty_native_pool_stays_empty_and_never_encodes_crops(self):
        self.native.predict.return_value = replace(
            self.result, boxes=np.empty((0, 4)), scores=np.empty(0), grounding_index=None,
            phrase_scores=np.empty(0), baseline_adjusted_scores=np.empty(0),
        )
        result = self.executor.predict(self.image, self.question)
        self.assertIsNone(result.grounding_index)
        self.assertEqual(len(result.boxes), 0)
        self.assertEqual(result.diagnostics["grounding_reason"], "grounding_empty_candidates")
        self.scorer.score.assert_not_called()
        self.convert.assert_not_called()

    def test_missing_same_forward_scores_falls_back_explicitly(self):
        for field in ("phrase_scores", "baseline_adjusted_scores"):
            with self.subTest(field=field):
                self.native.predict.return_value = replace(self.result, **{field: None})
                result = self.executor.predict(self.image, self.question)
                self.assertEqual(result.grounding_index, 0)
                self.assertEqual(result.diagnostics["grounding_reason"], "grounding_missing_scores")
        self.scorer.score.assert_not_called()

    def test_original_scorer_failure_keeps_this_forward_native_selection(self):
        self.scorer.score.side_effect = CropIdentityError("clip_image_encoding_failed")
        result = self.executor.predict(self.image, self.question)
        self.assertEqual(result.grounding_index, 0)
        self.assertEqual(result.diagnostics["grounding_outcome"], "native_fallback")
        self.assertEqual(result.diagnostics["grounding_reason"], "grounding_crop_failed")

    def test_unexpected_optional_ranking_error_falls_back_without_exception_text(self):
        self.scorer.score.side_effect = RuntimeError("secret description and crop coordinates")
        result = self.executor.predict(self.image, self.question)
        self.assertEqual(result.grounding_index, 0)
        self.assertEqual(result.diagnostics["grounding_reason"], "grounding_rank_failed")
        self.assertNotIn("secret", str(result.diagnostics))

    def test_native_errors_propagate_and_do_not_enter_optional_ranking(self):
        error = RuntimeError("native failed")
        self.native.predict.side_effect = error
        with self.assertRaises(RuntimeError) as context:
            self.executor.predict(self.image, self.question)
        self.assertIs(context.exception, error)
        self.scorer.score.assert_not_called()

    def test_invalid_ranked_index_cannot_invent_a_box_or_replace_native(self):
        for index in (-1, 2, None, True):
            with self.subTest(index=index):
                with patch("tasks.detection.bbox_grounding.rank_full_candidates", return_value={
                    "arms": {"E": {"selected_index": index, "status": "succeeded"}},
                }):
                    result = self.executor.predict(self.image, self.question)
                self.assertEqual(result.grounding_index, 0)
                self.assertEqual(result.diagnostics["grounding_reason"], "grounding_invalid_selection")

    def test_contrast_tie_has_zero_correction_and_retains_native(self):
        self.contrast = [5., 5.]
        self.native.predict.return_value = replace(
            self.result, grounding_index=1, baseline_adjusted_scores=np.array([1., 1.]),
        )
        result = self.executor.predict(self.image, self.question)
        self.assertEqual(result.grounding_index, 1)

    def test_strong_native_semantics_are_not_overridden_by_crop_identity(self):
        self.native.predict.return_value = replace(self.result, baseline_adjusted_scores=np.array([1.3, 1.]))
        result = self.executor.predict(self.image, self.question)
        self.assertEqual(result.grounding_index, 0)

    def test_diagnostics_only_contain_fixed_enums_counts_and_time(self):
        result = self.executor.predict(self.image, self.question)
        diagnostic = result.diagnostics
        self.assertEqual(diagnostic["grounding_policy"], "clip_guarded")
        self.assertEqual(diagnostic["grounding_candidate_count"], 2)
        self.assertEqual(diagnostic["grounding_crop_count"], 2)
        self.assertEqual(diagnostic["nms_candidate_count"], 2)
        self.assertIsNot(diagnostic, self.result.diagnostics)
        for key in ("grounding_elapsed_ms", "grounding_candidate_count", "grounding_reference_count", "grounding_crop_count"):
            self.assertIs(type(diagnostic[key]), int)
            self.assertGreaterEqual(diagnostic[key], 0)
        for secret in (self.question, self.description, "boxes", "crop_scores", "selected_index"):
            self.assertNotIn(secret, str(diagnostic))

    def test_guarded_constructor_rejects_unfrozen_protocol_or_missing_dependencies(self):
        for confidence, spatial in ((.1, .15), (.25, 0), (.05, .15)):
            with self.subTest(confidence=confidence, spatial=spatial):
                native = Mock(candidate_confidence=confidence, spatial_weight=spatial)
                with self.assertRaises(ValueError):
                    BBoxGroundingExecutor(native, self.scorer, description_extractor=self.extract,
                                          image_converter=self.convert)
        with self.assertRaises(ValueError):
            BBoxGroundingExecutor(self.native, None, description_extractor=self.extract, image_converter=self.convert)
        with self.assertRaises(ValueError):
            BBoxGroundingExecutor(self.native, self.scorer, description_extractor=None, image_converter=self.convert)

    def test_mutated_native_protocol_is_rejected_before_parsing_or_inference(self):
        for field, value in (("candidate_confidence", .1), ("spatial_weight", 0)):
            with self.subTest(field=field):
                setattr(self.native, field, value)
                with self.assertRaises(ValueError):
                    self.executor.predict(self.image, self.question)
                setattr(self.native, field, .25 if field == "candidate_confidence" else .15)
        self.extract.assert_not_called()
        self.native.predict.assert_not_called()
        self.scorer.score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
