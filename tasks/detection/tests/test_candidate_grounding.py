"""Synthetic tests for one-view semantic scores and soft relation constraints."""

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.detection.candidate_grounding import predict_candidate_grounding
from tasks.detection.reference_plan import ReferencePlan, ReferenceSpec


def prediction(boxes, phrase=None, extra=None, index=0, confidence=None):
    return {"boxes": boxes, "phrase_scores": phrase, "extra_phrase_scores": extra,
            "grounding_index": index, "scores": confidence if confidence is not None else [.5] * len(boxes)}


class ScriptedDetector:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def predict(self, image, description, **kwargs):
        captured = dict(kwargs)
        if "score_boxes_xyxy" in captured:
            captured["score_boxes_xyxy"] = captured["score_boxes_xyxy"].copy()
        self.calls.append((image.copy(), description, captured))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(boxes=np.asarray(output["boxes"], dtype=float).reshape(-1, 4),
            scores=np.asarray(output["scores"], dtype=float), grounding_index=output["grounding_index"],
            phrase_scores=None if output["phrase_scores"] is None else np.asarray(output["phrase_scores"], dtype=float),
            extra_phrase_scores=None if output["extra_phrase_scores"] is None else np.asarray(output["extra_phrase_scores"], dtype=float),
            original_size=image.shape[:2])


def plan(refs=(ReferenceSpec("the tower", "left_of"),), absolute=None):
    return ReferencePlan("the red lawn left of the tower", "the red lawn", refs, absolute)


class CandidateGroundingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)

    def test_call_order_prompt_and_all_extra_boxes_match_one_final_forward(self):
        detector = ScriptedDetector([
            prediction([[100, 20, 120, 40]], [2]),
            prediction([[10, 10, 20, 20], [80, 10, 90, 20]], [100, -100]),
            prediction([[30, 40, 40, 50]], [1000]),
            prediction([[150, 40, 160, 50]], [2], [3, 1, 4]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual([x["kind"] for x in result["call_order"]],
                         ["reference", "target_roi", "target_full_original", "target_full_subject"])
        self.assertEqual([desc for _, desc, _ in detector.calls],
                         ["the tower", "the red lawn", "the red lawn left of the tower", "the red lawn"])
        self.assertTrue(all(kwargs["return_phrase_scores"] for _, _, kwargs in detector.calls))
        np.testing.assert_array_equal(detector.calls[-1][2]["score_boxes_xyxy"],
                                      [[10, 10, 20, 20], [80, 10, 90, 20], [30, 40, 40, 50]])
        self.assertEqual(result["full_subject"]["extra_candidate_sources"], [["roi", 0], ["roi", 1], ["full_original", 0]])
        self.assertEqual(result["selected_box"], [30., 40., 40., 50.])
        self.assertEqual(result["selected_source"], "full_original")
        self.assertTrue(all(x["scoring_origin"] == "full_subject_same_forward" for x in result["pool"]))

    def test_low_confidence_higher_semantics_wins(self):
        detector = ScriptedDetector([
            prediction([], [], index=None), prediction([], [], index=None),
            prediction([[20, 20, 30, 30], [150, 70, 160, 80]], [-5, -2], [], index=0, confidence=[.99, .01]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["selected_box"], [150., 70., 160., 80.])
        self.assertEqual(result["full_subject"]["selected_index"], 0)
        self.assertEqual([x["semantic_norm"] for x in result["pool"]], [0., 1.])

    def test_all_roi_candidates_can_win_instead_of_only_roi_match(self):
        detector = ScriptedDetector([
            prediction([[100, 20, 120, 40]], [2]),
            prediction([[10, 10, 20, 20], [70, 10, 80, 20]], [100, 0], index=0),
            prediction([], [], index=None),
            prediction([[150, 40, 160, 50]], [1], [0, 10]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["roi_record"]["selected_index"], 0)
        self.assertEqual(result["selected_box"], [70., 10., 80., 20.])
        winner = result["pool"][result["selected_pool_index"]]
        self.assertEqual((winner["source"], winner["source_index"]), ("roi", 1))

    def test_reference_identity_uses_description_match_not_confidence(self):
        detector = ScriptedDetector([
            prediction([[140, 30, 160, 50], [40, 30, 60, 50]], [0, 2], index=1, confidence=[.99, .01]),
            prediction([], [], index=None), prediction([], [], index=None),
            prediction([[5, 10, 15, 20]], [1], []),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["references"][0]["selected_box"], [40., 30., 60., 50.])
        self.assertEqual(result["references"][0]["reliability"], 1.)
        self.assertEqual(result["roi_record"]["roi_xyxy"], [0, 0, 70, 100])

    def test_reference_margin_is_not_canceled_by_geometry_denominator(self):
        detector = ScriptedDetector([
            prediction([[20, 20, 30, 30], [150, 20, 160, 30], [100, 20, 120, 40]], [0, 99, 100], index=2),
            prediction([], [], index=None), prediction([], [], index=None),
            prediction([[50, 40, 60, 50]], [1], []),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertAlmostEqual(result["references"][0]["reliability"], .01)
        self.assertAlmostEqual(result["pool"][0]["geometry"], .01 * (.5 + (110 - 55) / 200))

    def test_multiple_references_absolute_and_xy_offset_are_global(self):
        refs = (ReferenceSpec("tower", "right_of"), ReferenceSpec("road", "below"))
        detector = ScriptedDetector([
            prediction([[40, 20, 60, 40]], [1]), prediction([[100, 40, 120, 60]], [1]),
            prediction([[20, 10, 40, 30]], [1]), prediction([], [], index=None),
            prediction([], [], [2], index=None),
        ])
        result = predict_candidate_grounding(detector, self.image, plan(refs, (1, 1)))
        self.assertEqual(result["roi_record"]["roi_xyxy"], [60, 40, 200, 100])
        self.assertEqual(result["selected_box"], [80., 50., 100., 70.])
        box = result["pool"][0]
        expected_abs = (.45 + .6) / 2
        expected_right = .5 + (90 - 50) / 200
        expected_below = .5 + (60 - 50) / 100
        self.assertAlmostEqual(box["geometry"], (expected_abs + expected_right + expected_below) / 3)
        np.testing.assert_array_equal(detector.calls[-1][2]["score_boxes_xyxy"], [[80, 50, 100, 70]])

    def test_internal_crop_fragment_is_suppressed_while_full_box_is_kept(self):
        detector = ScriptedDetector([
            prediction([[100, 20, 120, 40]], [1]),
            prediction([[100, 20, 130, 40]], [1]),
            prediction([[90, 20, 150, 40]], [1]),
            prediction([], [], [100, 1], index=None),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["selected_box"], [90., 20., 150., 40.])
        fragment = next(x for x in result["pool"] if x["source"] == "roi")
        self.assertTrue(fragment["suppressed"])
        self.assertEqual(fragment["suppression_reason"], "roi_internal_boundary_full_overlap")
        self.assertIsNone(fragment["semantic_norm"])
        self.assertEqual(result["active_candidate_count"], 1)
        self.assertEqual(result["pool_candidate_count"], 2)

    def test_original_image_edge_does_not_count_as_internal_crop_edge(self):
        detector = ScriptedDetector([
            prediction([[100, 20, 120, 40]], [1]), prediction([[0, 20, 30, 40]], [1]),
            prediction([[0, 20, 50, 40]], [1]), prediction([], [], [10, 0], index=None),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        fragment = next(x for x in result["pool"] if x["source"] == "roi")
        self.assertFalse(fragment["suppressed"])
        self.assertEqual(result["selected_box"], [0., 20., 30., 40.])

    def test_duplicate_priority_keeps_actual_subject_box_without_coordinate_union(self):
        detector = ScriptedDetector([
            prediction([[100, 20, 120, 40]], [1]), prediction([[10, 10, 30, 30]], [1]),
            prediction([[10, 10, 30, 30]], [1]), prediction([[10, 10, 30, 30]], [1], [1000, 100]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["selected_source"], "full_subject")
        self.assertEqual(result["selected_box"], [10., 10., 30., 30.])
        self.assertEqual([x["suppression_reason"] for x in result["pool"]], [None, "duplicate_iou", "duplicate_iou"])
        self.assertEqual(result["suppressed_candidate_count"], 2)
        self.assertEqual(result["pool"][0]["semantic_norm"], 0.)

    def test_missing_and_tied_references_do_not_constrain_candidates(self):
        refs = (ReferenceSpec("tower", "left_of"), ReferenceSpec("road", "above"))
        detector = ScriptedDetector([
            prediction([], [], index=None), prediction([[20, 20, 40, 40], [120, 20, 140, 40]], [2, 2]),
            prediction([[10, 10, 20, 20]], [1]),
            prediction([[150, 70, 160, 80]], [1], [1]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan(refs))
        self.assertEqual([r["reliability"] for r in result["references"]], [0., 0.])
        self.assertIsNone(result["roi_record"])
        self.assertEqual(result["roi_fallback_reason"], "reference_unreliable")
        self.assertTrue(all(x["geometry"] == 0 and not x["self_match"] for x in result["pool"]))
        self.assertEqual(result["selected_source"], "full_subject")

    def test_selected_reference_below_raw_best_is_unreliable(self):
        detector = ScriptedDetector([
            prediction([[20, 20, 40, 40], [120, 20, 140, 40]], [0, 1], index=0),
            prediction([], [], index=None), prediction([[150, 70, 160, 80]], [1], []),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["references"][0]["reliability"], 0.)
        self.assertEqual(result["references"][0]["reliability_reason"], "selected_not_raw_best")
        self.assertIsNone(result["roi_record"])

    def test_self_match_penalty_is_soft_and_near_uses_bbox_distance(self):
        detector = ScriptedDetector([
            prediction([[20, 20, 40, 40]], [1]), prediction([[20, 20, 40, 40]], [1]),
            prediction([], [], index=None), prediction([[80, 20, 100, 40]], [0], [10]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan((ReferenceSpec("tower", "near"),)))
        near = next(x for x in result["pool"] if x["source"] == "roi")
        far = next(x for x in result["pool"] if x["source"] == "full_subject")
        self.assertTrue(near["self_match"])
        self.assertFalse(near["suppressed"])
        self.assertEqual(near["self_match_penalty"], .15)
        self.assertEqual(result["selected_box"], [20., 20., 40., 40.])
        self.assertAlmostEqual(far["geometry"], 1 - 50 / (.25 * math.hypot(200, 100)))

    def test_equal_semantics_zero_normalize_and_relation_can_change_selection(self):
        detector = ScriptedDetector([
            prediction([[100, 20, 120, 40]], [1]), prediction([], [], index=None),
            prediction([[10, 10, 20, 20]], [7]), prediction([[160, 10, 170, 20]], [-3], [-3]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertTrue(all(x["semantic_norm"] == 0 for x in result["pool"]))
        self.assertEqual(result["unconstrained_selected_box"], [160., 10., 170., 20.])
        self.assertEqual(result["selected_box"], [10., 10., 20., 20.])

    def test_missing_phrase_scores_falls_back_to_subject_match_without_confidence(self):
        detector = ScriptedDetector([
            prediction([], [], index=None), prediction([[20, 20, 30, 30]], None),
            prediction([[100, 20, 110, 30], [150, 20, 160, 30]], None, None, index=1, confidence=[.99, .01]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["selected_box"], [150., 20., 160., 30.])
        self.assertEqual(result["selection_reason"], "full_subject_match_fallback")
        self.assertEqual(result["fallback_reason"], "phrase_scores_unavailable")
        self.assertIsNone(result["unconstrained_selected_box"])
        self.assertEqual(result["active_candidate_count"], 0)

    def test_missing_subject_prediction_falls_back_to_original_match(self):
        detector = ScriptedDetector([prediction([], [], index=None),
            prediction([[20, 20, 30, 30], [50, 50, 60, 60]], None, index=1), RuntimeError("not logged")])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["selected_box"], [50., 50., 60., 60.])
        self.assertEqual(result["selection_reason"], "full_original_match_fallback")
        self.assertNotIn("not logged", str(result))

    def test_invalid_extra_score_shape_does_not_mix_scores_from_previous_forward(self):
        detector = ScriptedDetector([
            prediction([], [], index=None), prediction([[20, 20, 30, 30]], [1000]),
            prediction([[100, 20, 110, 30]], [1], [2, 3]),
        ])
        result = predict_candidate_grounding(detector, self.image, plan())
        self.assertEqual(result["full_subject"]["extra_phrase_scores_status"], "invalid")
        original = next(x for x in result["pool"] if x["source"] == "full_original")
        self.assertIsNone(original["semantic_raw"])
        self.assertTrue(original["suppressed"])
        self.assertEqual(result["selected_source"], "full_subject")

    def test_conflict_skips_roi_but_geometry_stays_soft_and_json_safe(self):
        refs = (ReferenceSpec("tower", "left_of"), ReferenceSpec("road", "right_of"))
        detector = ScriptedDetector([
            prediction([[20, 20, 40, 40]], [1]), prediction([[160, 20, 180, 40]], [1]),
            prediction([], [], index=None), prediction([[90, 40, 110, 60]], [-1], []),
        ])
        result = predict_candidate_grounding(detector, self.image, plan(refs))
        self.assertIsNone(result["roi_record"])
        self.assertEqual(result["roi_fallback_reason"], "conflicting_search_region")
        self.assertEqual(result["selected_box"], [90., 40., 110., 60.])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_invalid_image_fails_before_any_model_call(self):
        detector = ScriptedDetector([])
        with self.assertRaises(ValueError):
            predict_candidate_grounding(detector, np.zeros((0, 200, 3)), plan())
        self.assertEqual(detector.calls, [])


if __name__ == "__main__":
    unittest.main()
