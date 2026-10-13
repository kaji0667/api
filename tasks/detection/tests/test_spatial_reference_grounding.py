"""Geometry and source-text tests for the requested ABC-before-target path."""
from dataclasses import dataclass, replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tasks.detection.spatial_reference_grounding import (
    SpatialReferenceGroundingExecutor, parse_spatial_reference_plan, positioned_reference,
    ReferenceEnclosureResult,
)
from tasks.detection.structured_spatial_parser import (
    ContainerSpec, PositionedReferenceSpec, SpatialReferencePlan, SpatialParseError,
)


@dataclass(frozen=True)
class Result:
    boxes: np.ndarray
    scores: np.ndarray
    original_size: tuple
    grounding_index: int | None
    model_prompt: str = "source"
    diagnostics: dict | None = None
    phrase_scores: np.ndarray | None = None
    baseline_adjusted_scores: np.ndarray | None = None


def result(boxes, hw=(100, 200), phrase=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    row = np.asarray(phrase if phrase is not None else [1.] * len(boxes), dtype=np.float32)
    return Result(boxes, np.linspace(.99, .26, len(boxes), dtype=np.float32), hw,
                  0 if len(boxes) else None, phrase_scores=row, baseline_adjusted_scores=row)


class SpatialReferenceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.native = Mock()
        self.executor = SpatialReferenceGroundingExecutor(
            self.native, description_extractor=lambda text: text, image_converter=lambda image: image,
        )

    def test_source_queries_carry_position_and_keep_attributes(self):
        for text, target, prefix in (
            ("左下角的梯形草坪，右侧有道路，上方有白色建筑，旁边有池塘", "梯形草坪", "左下角的"),
            ("The lower-left trapezoidal lawn, left of a road, below a white building, near a pond",
             "The trapezoidal lawn", None),
            ("The lower-left lawn. It is trapezoidal. It is left of a road. A building is above it.",
             "The lawn, trapezoidal", None),
            ("左下角的草坪。它呈梯形。它在道路左侧。上方有白色建筑。", "草坪, 梯形", "左下角的"),
            ("The trapezoidal lawn. It is in the lower-left corner of the image. It is left of a road.",
             "The trapezoidal lawn", None),
        ):
            with self.subTest(text=text):
                plan = parse_spatial_reference_plan(text)
                self.assertIsNotNone(plan)
                self.assertEqual(plan.target_query, target)
                self.assertEqual(plan.absolute_position, (-1, 1))
                for spec in plan.references:
                    query, position = positioned_reference(spec, plan.absolute_position)
                    self.assertEqual(position, (-1, 1))
                    if prefix:
                        self.assertEqual(query, prefix + spec.description)
                    else:
                        self.assertEqual(query, spec.description + " in the bottom-left corner of the image")

    def test_reference_explicit_location_is_not_overwritten(self):
        plan = parse_spatial_reference_plan("The lower-left lawn, below a building in the top-left corner")
        self.assertEqual(positioned_reference(plan.references[0], plan.absolute_position),
                         ("a building in the top-left corner", (-1, -1)))

    def test_all_abc_precede_target_match_is_not_confidence_and_coordinates_are_restored(self):
        question = "左下角的梯形草坪，右侧有道路，上方有建筑，旁边有池塘"
        references = [result([[5, 5, 15, 15], [85, 55, 95, 85]], phrase=[10, 1]),
                      result([[20, 40, 70, 50]]), result([[10, 50, 40, 80]])]
        calls = []

        def predict(image, text, **kwargs):
            calls.append((text, image.shape[:2], kwargs))
            if len(calls) <= 3:
                return references[len(calls) - 1]
            self.assertEqual(text, "梯形草坪")
            self.assertEqual(kwargs, {"require_grounding": False, "return_phrase_scores": True})
            # ROI origin is (0,35); the first candidate is outside the derived
            # region despite having the largest phrase score and confidence.
            return result([[75, 0, 95, 10], [20, 20, 50, 45], [30, 25, 60, 50]],
                          hw=image.shape[:2], phrase=[20, 2, 5])

        self.native.predict.side_effect = predict
        prediction = self.executor.predict(self.image, question)
        self.assertEqual([call[0] for call in calls], ["左下角的道路", "左下角的建筑", "左下角的池塘", "梯形草坪"])
        self.assertEqual(prediction.grounding_index, 2)
        np.testing.assert_array_equal(prediction.boxes[2], [30, 60, 60, 85])
        self.assertEqual(prediction.original_size, (100, 200))
        self.assertEqual(prediction.diagnostics["grounding_outcome"], "region_selected")
        self.assertEqual(prediction.diagnostics["grounding_located_reference_count"], 3)
        self.assertEqual(prediction.diagnostics["grounding_detector_call_count"], 4)

    def test_nonzero_xy_roi_origin_and_full_box_extent(self):
        question = "The upper-right lawn, right of a road, above a building"
        calls = 0
        def predict(image, text, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return result([[90, 10, 110, 30]])
            if calls == 2:
                return result([[120, 40, 180, 60]])
            return result([[40, 5, 90, 35]], hw=image.shape[:2])
        self.native.predict.side_effect = predict
        prediction = self.executor.predict(self.image, question)
        np.testing.assert_array_equal(prediction.boxes[0], [120, 5, 170, 35])
        self.assertEqual(prediction.original_size, (100, 200))

    def test_direct_and_reverse_relations_give_the_same_plan(self):
        a = parse_spatial_reference_plan("The lower-left lawn, left of a road, below a building")
        b = parse_spatial_reference_plan("The lower-left lawn, a road is right of it, a building is above it")
        self.assertEqual(a, b)

    def test_unsupported_keeps_entire_original_question(self):
        for text in ("草坪，不在道路左侧", "The lawn between a road and a building", "lawn, unexplained sentence"):
            with self.subTest(text=text):
                self.native.reset_mock()
                self.native.predict.return_value = result([[1, 2, 20, 30]])
                output = self.executor.predict(self.image, text)
                self.native.predict.assert_called_once_with(self.image, text)
                self.assertEqual(output.diagnostics["grounding_parse_status"], "unsupported")

    def test_missing_reference_and_conflicting_geometry_fall_back_without_roi(self):
        for values, reason in (
            ([result([]), result([[10, 40, 20, 60]])], "grounding_reference_unavailable"),
            ([result([[80, 40, 100, 60]]), result([[10, 40, 20, 60]])], "grounding_conflicting_region"),
        ):
            with self.subTest(reason=reason):
                self.native.reset_mock()
                baseline = result([[20, 60, 40, 80]])
                self.native.predict.side_effect = [*values, baseline]
                question = "The lower-left lawn, right of a road, left of a building"
                output = self.executor.predict(self.image, question)
                self.assertEqual(output.diagnostics["grounding_reason"], reason)
                self.assertEqual(output.diagnostics["grounding_roi_count"], 0)
                self.assertEqual(self.native.predict.call_args.args[1], question)
                np.testing.assert_array_equal(output.boxes, baseline.boxes)

    def test_no_anchor_or_roi_box_is_fabricated_as_target(self):
        for target in (result([]), result([[10, 0, 50, 15]])):
            with self.subTest(boxes=target.boxes.tolist()):
                self.native.reset_mock()
                question = "The lower-left lawn, near a building"
                anchor = result([[10, 50, 50, 65]])
                baseline = result([[70, 60, 90, 80]])
                def predict(image, text, **kwargs):
                    if text == question:
                        return baseline
                    if text.startswith("a building"):
                        return anchor
                    return replace(target, original_size=image.shape[:2])
                self.native.predict.side_effect = predict
                output = self.executor.predict(self.image, question)
                np.testing.assert_array_equal(output.boxes, baseline.boxes)
                self.assertEqual(output.diagnostics["grounding_outcome"], "native_fallback")

    def test_invalid_native_scores_and_error_text_are_not_logged(self):
        question = "The lower-left lawn, near a road"
        bad = replace(result([[10, 50, 20, 60]]), baseline_adjusted_scores=np.array([np.nan]))
        baseline = result([[20, 60, 40, 80]])
        self.native.predict.side_effect = [bad, baseline]
        output = self.executor.predict(self.image, question)
        self.assertEqual(output.diagnostics["grounding_outcome"], "native_fallback")
        self.assertNotIn("lawn", str(output.diagnostics))


class StructuredExecutorTests(unittest.TestCase):
    setUp = SpatialReferenceTests.setUp

    def use(self, plan):
        self.parser = Mock(return_value=plan, name="parser")
        self.parser.name = "qwen"
        self.executor = SpatialReferenceGroundingExecutor(
            self.native, description_extractor=lambda text: text, image_converter=lambda image: image,
            plan_parser=self.parser,
        )

    def test_structured_explicit_reference_position_is_used_in_query_and_geometry(self):
        spec = PositionedReferenceSpec("a white tower", "below", (1, -1), ("below",))
        self.assertEqual(positioned_reference(spec, (-1, 1)),
                         ("a white tower in the top-right corner of the image", (1, -1)))

    def test_four_references_are_detected_before_target(self):
        self.use(SpatialReferencePlan("a blue shed", (
            PositionedReferenceSpec("a lane", "below"),
            PositionedReferenceSpec("a yard", "right_of"),
            PositionedReferenceSpec("a lawn", "above"),
            PositionedReferenceSpec("two tanks", "left_of"),
        ), (-1, 1)))
        values = [result([[20, 35, 80, 45]]), result([[0, 50, 10, 80]]),
                  result([[10, 85, 90, 95]]), result([[95, 50, 105, 80]])]
        def predict(image, text, **kwargs):
            if values:
                return values.pop(0)
            return result([[30, 30, 60, 50]], hw=image.shape[:2])
        self.native.predict.side_effect = predict
        output = self.executor.predict(self.image, "original question")
        self.assertEqual([c.args[1] for c in self.native.predict.call_args_list],
                         ["a lane in the bottom-left corner of the image",
                          "a yard in the bottom-left corner of the image",
                          "a lawn in the bottom-left corner of the image",
                          "two tanks in the bottom-left corner of the image", "a blue shed"])
        self.assertEqual(output.diagnostics["grounding_outcome"], "region_selected")
        self.assertEqual(output.diagnostics["grounding_located_reference_count"], 4)

    def test_container_local_upper_left_does_not_become_image_upper_left(self):
        self.use(SpatialReferencePlan("the only yellow van, a black roof", (), None,
                                     (ContainerSpec("a parking lot", (1, None), (-1, -1)),)))
        def predict(image, text, **kwargs):
            if text.startswith("a parking lot"):
                return result([[110, 40, 190, 90]])
            self.assertEqual(text, "the only yellow van, a black roof")
            return result([[30, 20, 45, 30]], hw=image.shape[:2])
        self.native.predict.side_effect = predict
        output = self.executor.predict(self.image, "original question")
        # ROI origin (90,30); container-local quadrant remains on image RIGHT.
        np.testing.assert_array_equal(output.boxes, [[120, 50, 135, 60]])
        self.assertEqual(output.diagnostics["grounding_outcome"], "region_selected")
        self.assertEqual(output.diagnostics["grounding_located_container_count"], 1)
        self.assertEqual(output.diagnostics["grounding_detector_call_count"], 2)

    def test_frame_edge_is_not_queried_as_an_object(self):
        self.use(SpatialReferencePlan("a red hut", (), (-1, None), frame_edges=("left",)))
        self.native.predict.return_value = result([[10, 20, 30, 40]], hw=(100, 140))
        output = self.executor.predict(self.image, "original question")
        self.assertEqual(self.native.predict.call_args.args[1], "a red hut")
        self.assertEqual(output.diagnostics["grounding_reference_count"], 0)
        self.assertEqual(output.diagnostics["grounding_detector_call_count"], 1)
        self.assertEqual(output.diagnostics["grounding_outcome"], "region_selected")

    def test_parser_failure_uses_original_native_and_fixed_reason(self):
        self.use(None)
        self.parser.side_effect = SpatialParseError("grounding_parser_source_mismatch")
        self.native.predict.return_value = result([[1, 2, 20, 30]])
        output = self.executor.predict(self.image, "original question")
        self.native.predict.assert_called_once_with(self.image, "original question")
        self.assertEqual(output.diagnostics["grounding_parser"], "qwen")
        self.assertEqual(output.diagnostics["grounding_reason"], "grounding_parser_source_mismatch")
        self.assertEqual(output.diagnostics["grounding_roi_count"], 0)

    def test_simple_query_is_parsed_and_goes_native(self):
        self.use(SpatialReferencePlan("sheds", (), None))
        self.native.predict.return_value = result([[1, 2, 20, 30]])
        output = self.executor.predict(self.image, "sheds")
        self.native.predict.assert_called_once_with(self.image, "sheds")
        self.assertEqual(output.diagnostics["grounding_parse_status"], "parsed")
        self.assertEqual(output.diagnostics["grounding_reason"], "grounding_no_spatial_constraints")


class ReferenceEnclosureTests(unittest.TestCase):
    setUp = SpatialReferenceTests.setUp

    def use(self, references, **fields):
        plan = SpatialReferencePlan("target identity", tuple(references), None, **fields)
        self.executor = SpatialReferenceGroundingExecutor(
            self.native, description_extractor=lambda text: text, image_converter=lambda image: image,
            plan_parser=lambda text: plan, enclose_references=True,
        )

    def test_all_reference_extents_return_directly_without_roi_or_target_detection(self):
        self.use([PositionedReferenceSpec("a lane", "below"),
                  PositionedReferenceSpec("a yard", "right_of"),
                  PositionedReferenceSpec("a lawn", "above"),
                  PositionedReferenceSpec("two tanks", "left_of")])
        self.native.predict.side_effect = [
            result([[20.5, 5.2, 90, 20]]), result([[2.3, 30, 18, 70]]),
            result([[30, 75, 80, 95.4]]), result([[100, 20, 150.8, 65]]),
        ]
        with patch("tasks.detection.spatial_reference_grounding._expanded_roi", side_effect=AssertionError), \
                patch("tasks.detection.spatial_reference_grounding._relation_region", side_effect=AssertionError):
            output = self.executor.predict(self.image, "original question")
        self.assertIsInstance(output, ReferenceEnclosureResult)
        np.testing.assert_allclose(output.bbox, [2.3, 5.2, 150.8, 95.4], atol=1e-5)
        self.assertEqual(output.original_size, (100, 200))
        self.assertFalse(hasattr(output, "scores"))
        self.assertEqual([call.args[1] for call in self.native.predict.call_args_list],
                         ["a lane", "a yard", "a lawn", "two tanks"])
        for call in self.native.predict.call_args_list:
            self.assertIs(call.args[0], self.image)
            self.assertEqual(call.kwargs, {"return_phrase_scores": True})
        self.assertEqual(output.diagnostics["grounding_policy"], "reference_enclosure")
        self.assertEqual(output.diagnostics["grounding_outcome"], "enclosure_selected")
        self.assertEqual(output.diagnostics["grounding_reason"], "grounding_reference_enclosed")
        self.assertEqual(output.diagnostics["grounding_detector_call_count"], 4)
        self.assertEqual(output.diagnostics["grounding_roi_count"], 0)
        self.assertEqual(output.diagnostics["grounding_candidate_count"], 0)

    def test_overlapping_opposing_references_do_not_collapse_to_center(self):
        self.use([PositionedReferenceSpec("road", "below"), PositionedReferenceSpec("yard", "above")])
        self.native.predict.side_effect = [result([[20, 30, 80, 90]]), result([[20.01, 29.99, 80.02, 90]])]
        output = self.executor.predict(self.image, "original question")
        np.testing.assert_allclose(output.bbox, [20, 29.99, 80.02, 90], atol=1e-5)
        self.assertEqual(self.native.predict.call_count, 2)

    def test_single_reference_keeps_full_extent(self):
        self.use([PositionedReferenceSpec("road", "near")])
        self.native.predict.return_value = result([[140, 60, 200, 100]])
        output = self.executor.predict(self.image, "original question")
        self.assertEqual(output.bbox, (140, 60, 200, 100))
        self.native.predict.assert_called_once_with(self.image, "road", return_phrase_scores=True)

    def test_missing_reference_falls_back_without_partial_enclosure_or_crop(self):
        self.use([PositionedReferenceSpec("road", "below"), PositionedReferenceSpec("yard", "above")])
        baseline = result([[40, 50, 60, 70]])
        self.native.predict.side_effect = [result([[20, 30, 80, 90]]), result([]), baseline]
        output = self.executor.predict(self.image, "original question")
        self.assertNotIsInstance(output, ReferenceEnclosureResult)
        np.testing.assert_array_equal(output.boxes, baseline.boxes)
        self.native.predict.assert_called_with(self.image, "original question")
        self.assertEqual(output.diagnostics["grounding_located_reference_count"], 1)
        self.assertEqual(output.diagnostics["grounding_roi_count"], 0)
        self.assertEqual(output.diagnostics["grounding_reason"], "grounding_reference_unavailable")

    def test_all_references_attempted_after_empty_unmatched_and_detection_error(self):
        self.use([PositionedReferenceSpec("road", "below"),
                  PositionedReferenceSpec("yard", "above", (-1, -1)),
                  PositionedReferenceSpec("trees", "near"),
                  PositionedReferenceSpec("pond", "near")])
        empty = replace(result([]), diagnostics={"max_confidence": .012})
        unmatched = replace(result([[150, 80, 180, 95]]), diagnostics={"max_confidence": .9})
        accepted = replace(result([[20, 30, 80, 90]]), diagnostics={"max_confidence": .8})
        baseline = replace(result([[40, 50, 60, 70]]), diagnostics={"max_confidence": .99})
        self.native.predict.side_effect = [empty, unmatched, RuntimeError("private-error-text"), accepted, baseline]
        with patch("tasks.detection.spatial_reference_grounding._expanded_roi", side_effect=AssertionError):
            output = self.executor.predict(self.image, "original question")
        self.assertNotIsInstance(output, ReferenceEnclosureResult)
        np.testing.assert_array_equal(output.boxes, baseline.boxes)
        self.assertEqual([call.args[1] for call in self.native.predict.call_args_list],
                         ["road", "yard in the top-left corner of the image", "trees", "pond", "original question"])
        attempts = output.diagnostics["grounding_reference_results"]
        self.assertEqual([attempt["reference_index"] for attempt in attempts], [1, 2, 3, 4])
        self.assertEqual([attempt["status"] for attempt in attempts],
                         ["no_candidates", "match_failed", "error", "selected"])
        self.assertEqual([attempt["candidate_count"] for attempt in attempts], [0, 1, None, 1])
        self.assertEqual([attempt["max_confidence"] for attempt in attempts], [.012, .9, None, .8])
        self.assertEqual(attempts[2]["failure_stage"], "detection")
        self.assertEqual(attempts[3]["selected_index"], 0)
        self.assertEqual(attempts[3]["selected_bbox_original_xyxy"], [20, 30, 80, 90])
        self.assertEqual(attempts[0]["query"], "road")
        self.assertIsNone(attempts[0]["selected_bbox_original_xyxy"])
        self.assertNotIn("private-error-text", str(output.diagnostics))
        self.assertEqual(output.diagnostics["grounding_located_reference_count"], 1)
        self.assertEqual(output.diagnostics["grounding_detector_call_count"], 5)
        self.assertEqual(output.diagnostics["grounding_reason"], "grounding_reference_unavailable")
        self.assertEqual(output.diagnostics["grounding_roi_count"], 0)

    def test_candidate_and_matcher_exceptions_do_not_skip_later_references(self):
        for invalid, phase in (
            (replace(result([[20, 30, 80, 90]]), original_size=(90, 200)), "candidates"),
            (replace(result([[20, 30, 80, 90]]), baseline_adjusted_scores=np.array([np.nan])), "matching"),
        ):
            with self.subTest(phase=phase):
                self.native.reset_mock()
                self.use([PositionedReferenceSpec("road", "below"), PositionedReferenceSpec("yard", "above")])
                baseline = result([[40, 50, 60, 70]])
                self.native.predict.side_effect = [invalid, result([[20, 30, 80, 90]]), baseline]
                output = self.executor.predict(self.image, "original question")
                attempts = output.diagnostics["grounding_reference_results"]
                self.assertEqual([attempt["status"] for attempt in attempts], ["error", "selected"])
                self.assertEqual(attempts[0]["failure_stage"], phase)
                self.assertEqual(self.native.predict.call_count, 3)
                np.testing.assert_array_equal(output.boxes, baseline.boxes)

    def test_failed_container_still_observes_all_references_and_preserves_fallback(self):
        for unavailable in (result([]), RuntimeError("private-container-error")):
            with self.subTest(unavailable=type(unavailable).__name__):
                self.native.reset_mock()
                self.use([PositionedReferenceSpec("road", "below"), PositionedReferenceSpec("yard", "above")],
                         containers=(ContainerSpec("a park", None, (-1, -1)),))
                baseline = result([[40, 50, 60, 70]])
                self.native.predict.side_effect = [unavailable, result([[10, 20, 30, 40]]),
                                                   result([[40, 50, 80, 90]]), baseline]
                output = self.executor.predict(self.image, "original question")
                self.assertEqual(output.diagnostics["grounding_located_reference_count"], 2)
                self.assertEqual(output.diagnostics["grounding_detector_call_count"], 4)
                self.assertEqual(output.diagnostics["grounding_reason"], "grounding_container_unavailable")
                self.assertNotIsInstance(output, ReferenceEnclosureResult)
                np.testing.assert_array_equal(output.boxes, baseline.boxes)

    def test_final_original_description_fallback_error_still_propagates(self):
        self.use([PositionedReferenceSpec("road", "below"), PositionedReferenceSpec("yard", "above")])
        failure = RuntimeError("fallback error")
        self.native.predict.side_effect = [result([]), result([[20, 30, 80, 90]]), failure]
        with self.assertRaises(RuntimeError) as caught:
            self.executor.predict(self.image, "original question")
        self.assertIs(caught.exception, failure)
        self.native.predict.assert_called_with(self.image, "original question")
        self.assertEqual(self.native.predict.call_count, 3)

    def test_no_entities_does_not_enclose_container_or_frame_or_crop_target(self):
        for fields in ({}, {"frame_edges": ("left",)},
                       {"containers": (ContainerSpec("a parking lot", None, (-1, -1)),)}):
            with self.subTest(fields=fields):
                self.native.reset_mock()
                self.use([], **fields)
                self.native.predict.return_value = result([[40, 50, 60, 70]])
                output = self.executor.predict(self.image, "original question")
                self.native.predict.assert_called_once_with(self.image, "original question")
                self.assertEqual(output.diagnostics["grounding_reason"], "grounding_no_references")
                self.assertEqual(output.diagnostics["grounding_roi_count"], 0)

    def test_parser_rejection_preserves_original_native_fallback(self):
        parser = Mock(side_effect=SpatialParseError("grounding_parser_unresolved"))
        self.executor = SpatialReferenceGroundingExecutor(
            self.native, description_extractor=lambda text: text, image_converter=lambda image: image,
            plan_parser=parser, enclose_references=True,
        )
        self.native.predict.return_value = result([[40, 50, 60, 70]])
        output = self.executor.predict(self.image, "original question")
        self.native.predict.assert_called_once_with(self.image, "original question")
        self.assertEqual(output.diagnostics["grounding_reason"], "grounding_parser_unresolved")
        self.assertEqual(output.diagnostics["grounding_policy"], "reference_enclosure")


if __name__ == "__main__":
    unittest.main()
