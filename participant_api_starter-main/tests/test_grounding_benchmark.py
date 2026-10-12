"""Check diagnostic metrics, failure denominators and reproducible sampling."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_grounding import aggregate, box_iou, detection_parity, mask_box, outward_box, select_rows
from native_parity import _grounding_selection_parity


class GroundingDiagnosticTests(unittest.TestCase):
    def test_xyxy_iou_and_missing_predictions(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(box_iou(None, [0, 0, 10, 10]), 0)

    def test_mask_bbox_includes_last_foreground_pixel(self):
        mask = np.zeros((9, 13), dtype=bool)
        mask[3, 7] = True
        self.assertEqual(mask_box(mask), [7, 3, 8, 4])
        self.assertIsNone(mask_box(np.zeros_like(mask)))

    def test_oracle_uses_same_outward_rounding_as_selected_answer(self):
        candidate, target = [.1, 0, 10, 10], [0, 0, 5, 10]
        self.assertLess(box_iou(candidate, target), .5)
        self.assertEqual(outward_box(candidate), [0, 0, 10, 10])
        self.assertEqual(box_iou(outward_box(candidate), target), .5)
        self.assertIsNone(outward_box([3, 0, 3, 5]))

    def test_stratified_sampling_independent_of_metadata_order(self):
        rows = [{"id": f"val_{i}", "class_idx": i // 5} for i in range(15)]
        selected = select_rows(rows, 2, 42)
        self.assertEqual(selected, select_rows(list(reversed(rows)), 2, 42))
        self.assertEqual(len(selected), 6)
        self.assertEqual([row["class_idx"] for row in selected], [0, 0, 1, 1, 2, 2])
        with self.assertRaises(ValueError):
            select_rows(rows, 6, 42)

    def test_candidate_and_selection_failures_are_distinct(self):
        def result(iou, oracle, valid=True):
            return {"detection": {"box": [1, 1, 3, 3] if valid else None, "iou": iou,
                                  "mask_box_iou": iou, "oracle_iou": oracle, "latency_ms": 5}}
        # One success, one selectable-but-wrong choice, one absent candidate.
        metrics = aggregate([result(.8, .9), result(.1, .8), result(0, 0, False)], "detection")
        self.assertEqual(metrics["valid_count"], 2)
        self.assertAlmostEqual(metrics["acc_at_0.5"], 1 / 3)
        self.assertAlmostEqual(metrics["candidate_recall_at_0_5"], 2 / 3)
        self.assertEqual(metrics["selection_misses_at_0_5"], 1)
        self.assertEqual(metrics["candidate_misses_at_0_5"], 1)


class GroundingSelectionParityTests(unittest.TestCase):
    def fixture(self, weight, selected):
        # Rect padding is significant on this short, wide original image.
        tensor = torch.zeros((1, 3, 128, 832))
        boxes = np.array([[0., 10., 160., 40.], [640., 10., 800., 40.]], dtype=np.float32)
        network_boxes = torch.from_numpy(boxes) + torch.tensor([16., 14., 16., 14.])
        maps = [torch.tensor([[[[.48, .5]]]])]
        raw = torch.zeros((1, 5, 2))
        vectors = torch.ones((1, 1, 768))
        reference = {
            "network_boxes": network_boxes, "geometry_boxes": boxes,
            "boxes": torch.from_numpy(boxes), "scores": torch.tensor([.9, .2]),
            "phrase_scores": torch.tensor([[.48, .5]]), "selected": 1,
            "raw": raw, "maps": maps, "conditions": {"txt_vec": vectors},
        }
        observed = {"image": tensor, "raw": raw, "phrase_maps": maps,
                    "conditions": {"txt_vec": vectors}}
        prediction = SimpleNamespace(boxes=boxes, scores=np.array([.9, .2], dtype=np.float32),
                                     original_size=(100, 800), grounding_index=selected)
        native = SimpleNamespace(backend=SimpleNamespace(stride=32), spatial_weight=weight,
                                 image_size_for=lambda shape: 800)
        return native, prediction, reference, observed, tensor

    def parity(self, weight, selected, hint):
        native, prediction, reference, observed, tensor = self.fixture(weight, selected)
        with patch("benchmark_grounding._original_detection_image",
                   return_value=(tensor, ((1., 1.), (16., 14.)), (100, 800))), \
                patch("benchmark_grounding._detection_reference", return_value=reference), \
                patch("grounding_spatial.parse_absolute_position", return_value=hint) as parse:
            result = detection_parity(native, Path("public-synthetic.jpg"), "a target at the left",
                                      prediction, observed)
        return result, parse

    def test_enabled_bonus_changes_selection_without_breaking_model_parity(self):
        result, _ = self.parity(.15, 0, SimpleNamespace(horizontal=-1, vertical=None))
        self.assertEqual(result["original_selected_index"], 1)
        self.assertEqual(result["expected_selected_index"], 0)
        self.assertEqual(result["native_selected_index"], 0)
        self.assertFalse(result["assignment_equal"])
        self.assertTrue(result["selection_policy_checked"])
        self.assertEqual(result["semantic_scores_max_abs_error"], 0)
        self.assertEqual(result["restored_boxes_max_abs_error"], 0)
        self.assertEqual(result["raw_max_abs_error"], 0)

    def test_weight_zero_retains_strict_original_assignment_without_parsing(self):
        result, parse = self.parity(0, 1, SimpleNamespace(horizontal=-1, vertical=None))
        parse.assert_not_called()
        self.assertTrue(result["assignment_equal"])
        self.assertIsNone(result["spatial_hint"])
        with self.assertRaisesRegex(AssertionError, "detection_spatial_selection"):
            self.parity(0, 0, None)

    def test_no_hint_retains_original_assignment(self):
        result, _ = self.parity(.15, 1, None)
        self.assertTrue(result["assignment_equal"])
        with self.assertRaisesRegex(AssertionError, "detection_spatial_selection"):
            self.parity(.15, 0, None)

    def test_center_prior_cannot_promote_an_empty_box(self):
        native, prediction, reference, _, _ = self.fixture(.15, 1)
        reference["geometry_boxes"] = np.array([[400., 50., 400., 50.],
                                               [320., 40., 480., 60.]], dtype=np.float32)
        prediction.boxes = reference["geometry_boxes"]
        reference["phrase_scores"] = torch.tensor([[.50, .49]])
        reference["selected"] = 0
        with patch("grounding_spatial.parse_absolute_position",
                   return_value=SimpleNamespace(horizontal=0, vertical=0)):
            result = _grounding_selection_parity(native, "a central target", prediction, reference)
        self.assertEqual(result["expected_selected_index"], 1)
        self.assertFalse(result["assignment_equal"])


if __name__ == "__main__":
    unittest.main()
