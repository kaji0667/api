"""Generic-region evidence routing and complete-grid geometry regressions."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from .answering import EvidenceAnswerer, generic_region_query, region_answer
from .inference import ChangeEvidence, ChangePredictor
from .regions import mask_region_pixels


NINE = ["top_left", "top", "top_right", "left", "center", "right",
        "bottom_left", "bottom", "bottom_right"]
CONSTRAINT = {"type": "enum", "question_form": "change_region", "values": NINE}


def binary_evidence(mask):
    return {"mask_scope": "generic_binary_change", "mask_grid": {"binary": {
        "changed_pixels": int(np.count_nonzero(mask)), "region_pixels": mask_region_pixels(mask)}}}


class GenericQueryTests(unittest.TestCase):
    def test_generic_dominant_queries_accept_public_language_variants(self):
        queries = (
            "Which region has the largest change area?",
            "Which quadrant shows the most significant change?",
            "Which area exhibits the greatest overall changes?",
            "Which region has changed the most?",
            "Given two co-registered satellite images (before and after), identify the region with the largest change. Return only the region label.",
            "Comparing the earlier and later images, which region contains the maximum area of change?",
            "对比前后两幅图像，哪个区域变化最明显？仅回答区域名称。",
        )
        for question in queries:
            with self.subTest(question=question):
                self.assertTrue(generic_region_query(question))

    def test_counts_direction_rates_specific_class_and_extra_conditions_reject(self):
        for question in (
            "Which region has the smallest change area?",
            "Which region has the most new buildings?",
            "Which region has the largest added change area?",
            "Which region has the highest change density?",
            "Which region has the largest water change area?",
            "Which region has the largest change area near the road?",
            "Which region has the largest change area? Consider only demolished buildings.",
            "哪个区域的变化比例最大？",
            "哪个区域的新增建筑最多？",
            "哪个区域变化最明显？只考虑水体。",
        ):
            with self.subTest(question=question):
                self.assertFalse(generic_region_query(question))


class RegionRevisionTests(unittest.TestCase):
    def test_all_nine_edge_labels_use_declared_grid_and_original_values(self):
        for index, value in enumerate(NINE):
            with self.subTest(index=index):
                mask = np.zeros((257, 263), dtype=np.uint8)
                row, col = divmod(index, 3)
                mask[(20, 110, 256)[row], (20, 110, 262)[col]] = 1
                result = region_answer(CONSTRAINT, None, binary_evidence(mask), "binary")
                self.assertTrue(result.supported)
                self.assertEqual(result.answer, value)

    def test_odd_partition_counts_cover_every_pixel_without_overlap(self):
        mask = np.ones((257, 263), dtype=np.uint8)
        counts = mask_region_pixels(mask)
        self.assertEqual(sum(counts[name] for name in counts if not name.startswith(("upper_", "lower_"))), mask.size)
        self.assertEqual(sum(counts[name] for name in ("upper_left", "upper_right", "lower_left", "lower_right")), mask.size)

    def test_edge_names_do_not_imply_halves_or_incomplete_partitions(self):
        for values in (["top", "bottom", "left", "right"], NINE[:-1], [*NINE[:-1], "top_center"]):
            result = region_answer({"type": "enum", "values": values}, None,
                                   binary_evidence(np.ones((256, 256))), "binary")
            self.assertFalse(result.supported)
            self.assertEqual(result.reason, "ambiguous_region")

    def test_generic_region_cannot_relabel_mci_foreground_as_binary_change(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:20, 10:20] = 2
        answerer = EvidenceAnswerer()
        result = answerer.answer("Which region has the largest change area?", CONSTRAINT,
                                 None, ChangeEvidence("Buildings appeared.", mask, 0, (256, 256)).summary())
        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "insufficient_evidence")

    def test_generic_binary_answer_does_not_call_caption_language_model(self):
        answerer = EvidenceAnswerer()
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[200:220, 100:120] = 1
        result = answerer.answer("Which region shows the most significant change?", CONSTRAINT,
                                 None, binary_evidence(mask))
        self.assertEqual(result.answer, "bottom")
        self.assertTrue(result.supported)

    def test_question_partition_cannot_be_overridden_by_complete_options(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:110, 5:15] = 1
        answerer = EvidenceAnswerer()
        for question, values in (
            ("Which quadrant shows the most significant change?", NINE),
            ("Which region shows the most significant change? Choose one of the four regions.", NINE),
            ("Which region shows the most significant change? Choose one of the nine regions.",
             ["Upper Left", "Upper Right", "Lower Left", "Lower Right"]),
        ):
            with self.subTest(question=question):
                result = answerer.answer(question, {"type": "enum", "values": values},
                                         None, binary_evidence(mask))
                self.assertFalse(result.supported)
                self.assertEqual(result.reason, "ambiguous_region")

    def test_no_change_and_ties_abstain(self):
        for mask in (np.zeros((256, 256), dtype=np.uint8), np.pad(np.ones((10, 10), dtype=np.uint8), ((10, 236), (10, 236)))):
            if mask.any():
                mask[100:110, 100:110] = 1
            result = EvidenceAnswerer().answer("Which region has changed the most?", CONSTRAINT,
                                               None, binary_evidence(mask))
            self.assertFalse(result.supported)
            self.assertEqual(result.reason, "ambiguous_region")


class BinaryDispatchTests(unittest.TestCase):
    def predictor(self):
        predictor = ChangePredictor.__new__(ChangePredictor)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        summary = binary_evidence(mask)
        predictor.binary_predictor = Mock()
        predictor.binary_predictor.evidence.return_value = SimpleNamespace(summary=lambda: summary)
        predictor.evidence = Mock(side_effect=AssertionError("MCI must not execute for generic regions"))
        predictor.answerer = EvidenceAnswerer()
        return predictor

    def test_generic_pair_uses_binary_predictor_and_keeps_image_order(self):
        predictor = self.predictor()
        result = predictor.predict("earlier.png", "later.png", "Which region has changed the most?",
                                   response_constraint=CONSTRAINT)
        self.assertEqual(result.answer, "top_left")
        predictor.binary_predictor.evidence.assert_called_once_with("earlier.png", "later.png")
        predictor.evidence.assert_not_called()

    def test_missing_binary_executor_abstains_without_mci_substitution(self):
        predictor = self.predictor()
        predictor.binary_predictor = None
        result = predictor.predict("earlier.png", "later.png", "Which region has changed the most?",
                                   response_constraint=CONSTRAINT)
        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "insufficient_evidence")
        predictor.evidence.assert_not_called()

    def test_generic_regions_require_declared_options_before_visual_inference(self):
        predictor = self.predictor()
        result = predictor.predict("earlier.png", "later.png", "Which region has changed the most?",
                                   response_constraint={"type": "short_text"})
        self.assertEqual((result.supported, result.reason), (False, "unsupported_constraint"))
        predictor.binary_predictor.evidence.assert_not_called()
        predictor.evidence.assert_not_called()


if __name__ == "__main__":
    unittest.main()
