"""Synthetic call order, matching and original-coordinate geometry checks."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.detection.reference_grounding import predict_reference_grounding
from tasks.detection.reference_plan import ReferencePlan, ReferenceSpec


class ScriptedDetector:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def predict(self, image, description):
        self.calls.append((image.copy(), description))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        boxes, scores, index = output
        return SimpleNamespace(boxes=np.array(boxes, dtype=float).reshape(-1, 4),
                               scores=np.array(scores, dtype=float), grounding_index=index,
                               original_size=image.shape[:2])


def output(boxes, index=0, scores=None):
    return boxes, ([.5] * len(boxes) if scores is None else scores), index


def plan(references=(ReferenceSpec("the tower", "left_of"),), absolute=None):
    return ReferencePlan("the lawn left of the tower", "the lawn", references, absolute)


class ReferenceGroundingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)

    def test_references_precede_roi_and_full_target_with_unchanged_prompt(self):
        detector = ScriptedDetector([output([[140, 30, 160, 50]]), output([[15, 10, 35, 30]]),
                                     output([[100, 60, 120, 80]])])
        result = predict_reference_grounding(detector, self.image, plan())
        self.assertEqual([call["kind"] for call in result["call_order"]],
                         ["reference", "target_roi", "target_full"])
        self.assertEqual([desc for _, desc in detector.calls],
                         ["the tower", "the lawn left of the tower", "the lawn left of the tower"])
        self.assertEqual(result["selected_source"], "roi")
        self.assertEqual(result["selected_box"], [15., 10., 35., 30.])

    def test_roi_restores_nonzero_origin_and_pixels(self):
        self.image[:, :, 0] = np.arange(200, dtype=np.uint8)
        detector = ScriptedDetector([output([[40, 20, 60, 40]]), output([[20, 10, 40, 30]]),
                                     output([[10, 50, 30, 70]])])
        result = predict_reference_grounding(detector, self.image,
                                             plan((ReferenceSpec("the tower", "right_of"),)))
        roi = result["rois"][0]["roi_xyxy"]
        self.assertEqual(roi, [30, 0, 200, 100])
        self.assertEqual(result["selected_box"], [50., 10., 70., 30.])
        self.assertEqual(int(detector.calls[1][0][0, 0, 0]), 30)

    def test_missing_reference_falls_back_after_all_reference_calls(self):
        detector = ScriptedDetector([output([], None), output([[120, 50, 140, 70]]),
                                     output([[5, 5, 15, 15]])])
        p = plan((ReferenceSpec("tower", "left_of"), ReferenceSpec("road", "above")))
        result = predict_reference_grounding(detector, self.image, p)
        self.assertEqual([x["kind"] for x in result["call_order"]], ["reference", "reference", "target_full"])
        self.assertEqual(result["fallback_reason"], "reference_unavailable")
        self.assertEqual(result["selected_box"], [5., 5., 15., 15.])

    def test_reference_and_target_follow_original_match_not_confidence(self):
        detector = ScriptedDetector([
            output([[150, 30, 170, 50], [40, 30, 60, 50]], 1, [.99, .1]),
            output([[10, 10, 20, 20], [60, 10, 65, 20]], 0, [.1, .99]),
            output([[5, 5, 15, 15], [30, 30, 40, 40]], 0, [.1, .99]),
        ])
        result = predict_reference_grounding(detector, self.image, plan())
        self.assertEqual(result["references"][0]["candidate_count"], 2)
        self.assertEqual(result["references"][0]["selected_index"], 1)
        self.assertEqual(result["rois"][0]["roi_xyxy"], [0, 0, 70, 100])
        self.assertEqual(result["selected_box"], [10., 10., 20., 20.])

    def test_conflicting_relations_skip_roi(self):
        detector = ScriptedDetector([output([[20, 10, 40, 30]]), output([[140, 10, 160, 30]]),
                                     output([[70, 40, 90, 60]])])
        result = predict_reference_grounding(detector, self.image,
                    plan((ReferenceSpec("tower", "left_of"), ReferenceSpec("road", "right_of"))))
        self.assertEqual(result["fallback_reason"], "conflicting_search_region")
        self.assertEqual(len(result["rois"]), 0)

    def test_relation_mismatch_keeps_full_match_without_hard_filter(self):
        detector = ScriptedDetector([output([[40, 20, 60, 40]]), output([[55, 10, 65, 30]]),
                                     output([[160, 60, 180, 80]])])
        result = predict_reference_grounding(detector, self.image, plan())
        self.assertEqual(result["fallback_reason"], "relation_mismatch")
        self.assertEqual(result["selected_box"], [160., 60., 180., 80.])
        self.assertEqual(result["selected_source"], "full")

    def test_near_reference_self_match_cannot_become_target(self):
        detector = ScriptedDetector([output([[20, 20, 40, 40]]), output([[20, 20, 40, 40]]),
                                     output([[100, 50, 130, 70]])])
        result = predict_reference_grounding(detector, self.image,
                                             plan((ReferenceSpec("tower", "near"),)))
        self.assertTrue(result["rois"][0]["self_match"])
        self.assertEqual(result["fallback_reason"], "reference_self_match")
        self.assertEqual(result["selected_source"], "full")

    def test_absolute_position_is_global_and_padding_retains_context(self):
        detector = ScriptedDetector([output([[100, 10, 120, 30]]), output([[10, 10, 30, 30]]),
                                     output([[100, 10, 120, 30]])])
        result = predict_reference_grounding(detector, self.image,
                                             plan((ReferenceSpec("tower", "below"),), (-1, 1)))
        self.assertEqual(result["rois"][0]["roi_xyxy"], [0, 30, 140, 100])
        self.assertEqual(result["selected_box"], [10., 40., 30., 60.])
        self.assertTrue(result["rois"][0]["valid_relations"])

    def test_minimum_crop_and_border_clipping(self):
        detector = ScriptedDetector([output([[0, 0, 2, 2]]), output([[0, 10, 1, 20]]),
                                     output([[50, 50, 60, 60]])])
        result = predict_reference_grounding(detector, self.image, plan())
        roi = result["rois"][0]["roi_xyxy"]
        self.assertEqual(roi, [0, 0, 50, 100])
        self.assertGreaterEqual(roi[2] - roi[0], .25 * self.image.shape[1])

    def test_whole_frame_search_does_not_repeat_target_forward(self):
        detector = ScriptedDetector([output([[0, 0, 200, 100]]), output([[10, 10, 20, 20]])])
        result = predict_reference_grounding(detector, self.image,
                                             plan((ReferenceSpec("tower", "near"),)))
        self.assertEqual(result["fallback_reason"], "whole_frame_roi")
        self.assertEqual([x["kind"] for x in result["call_order"]], ["reference", "target_full"])

    def test_prediction_failure_has_fixed_status_and_full_fallback(self):
        detector = ScriptedDetector([RuntimeError("private text must not appear"), output([[1, 2, 3, 4]])])
        result = predict_reference_grounding(detector, self.image, plan())
        self.assertEqual(result["references"][0]["status"], "prediction_failed")
        self.assertNotIn("private text", str(result))
        self.assertEqual(result["selected_box"], [1., 2., 3., 4.])

    def test_no_target_candidate_returns_none(self):
        detector = ScriptedDetector([output([[140, 30, 160, 50]]), output([], None), output([], None)])
        result = predict_reference_grounding(detector, self.image, plan())
        self.assertIsNone(result["selected_box"])
        self.assertEqual(result["selection_reason"], "no_valid_target")
        self.assertEqual(result["fallback_reason"], "roi_prediction_unavailable")

    def test_invalid_reference_match_does_not_default_to_first_box(self):
        detector = ScriptedDetector([output([[50, 10, 70, 30]], True), output([[1, 2, 3, 4]])])
        result = predict_reference_grounding(detector, self.image, plan())
        self.assertEqual(result["references"][0]["status"], "invalid_match")
        self.assertEqual(result["selected_source"], "full")

    def test_multiple_references_all_precede_target_and_constrain_one_roi(self):
        detector = ScriptedDetector([output([[20, 20, 40, 40]]), output([[100, 10, 120, 30]]),
                                     output([[40, 30, 60, 50]]), output([[1, 1, 2, 2]])])
        result = predict_reference_grounding(detector, self.image,
                    plan((ReferenceSpec("tower", "right_of"), ReferenceSpec("road", "below"))))
        self.assertEqual([x["kind"] for x in result["call_order"]],
                         ["reference", "reference", "target_roi", "target_full"])
        self.assertEqual(result["rois"][0]["roi_xyxy"], [10, 10, 200, 100])
        self.assertEqual(result["selected_box"], [50., 40., 70., 60.])
        self.assertTrue(all(r["target_relation_satisfied"] for r in result["references"]))
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_above_relation_uses_y_axis_and_keeps_all_roi_candidates(self):
        detector = ScriptedDetector([output([[80, 60, 100, 80]]),
                                     output([[10, -5, 30, 20], [150, 72, 190, 79]], 0),
                                     output([[20, 90, 30, 100]])])
        result = predict_reference_grounding(detector, self.image,
                                             plan((ReferenceSpec("tower", "above"),)))
        self.assertEqual(result["rois"][0]["roi_xyxy"], [0, 0, 200, 80])
        self.assertEqual(result["selected_box"], [10., 0., 30., 20.])
        self.assertEqual(result["rois"][0]["candidate_count"], 2)
        self.assertEqual(result["rois"][0]["candidate_boxes"][1], [150., 72., 190., 79.])

    def test_absolute_hint_conflict_with_reference_does_not_crop(self):
        detector = ScriptedDetector([output([[10, 10, 30, 30]]), output([[150, 10, 170, 30]])])
        result = predict_reference_grounding(detector, self.image, plan(absolute=(1, None)))
        self.assertEqual(result["fallback_reason"], "conflicting_search_region")
        self.assertEqual(result["selected_source"], "full")

    def test_invalid_image_fails_before_detector_calls(self):
        detector = ScriptedDetector([])
        with self.assertRaises(ValueError):
            predict_reference_grounding(detector, np.zeros((0, 20, 3)), plan())
        self.assertEqual(detector.calls, [])


if __name__ == "__main__":
    unittest.main()
