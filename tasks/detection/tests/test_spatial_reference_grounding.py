"""Geometry and source-text tests for the requested ABC-before-target path."""
from dataclasses import dataclass, replace
import unittest
from unittest.mock import Mock

import numpy as np

from tasks.detection.spatial_reference_grounding import (
    SpatialReferenceGroundingExecutor, parse_spatial_reference_plan, positioned_reference,
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
            ([result([])], "grounding_reference_unavailable"),
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


if __name__ == "__main__":
    unittest.main()
