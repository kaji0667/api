"""Check diagnostic metrics, failure denominators and reproducible sampling."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_grounding import aggregate, box_iou, mask_box, outward_box, select_rows


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


if __name__ == "__main__":
    unittest.main()
