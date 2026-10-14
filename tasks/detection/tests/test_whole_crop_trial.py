import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from tasks.detection.whole_crop_trial import square_pad, rank_whole_crop, select_whole_crop


class WholeCropTrialTests(unittest.TestCase):
    def test_padding_preserves_both_ends_and_does_not_distort_pixels(self):
        image = Image.new("RGB", (12, 2), (20, 30, 40))
        image.putpixel((0, 0), (255, 0, 0))
        image.putpixel((11, 1), (0, 255, 0))
        padded = square_pad(image, (1, 2, 3))
        self.assertEqual(padded.size, (12, 12))
        self.assertEqual(padded.getpixel((0, 5)), (255, 0, 0))
        self.assertEqual(padded.getpixel((11, 6)), (0, 255, 0))
        self.assertEqual(padded.getpixel((6, 0)), (1, 2, 3))

    def test_unknown_role_grammar_is_scored_verbatim_with_no_fake_references(self):
        scorer = Mock()
        scorer.score.return_value = {"records": [
            {"index": 0, "status": "succeeded", "contrast_score": .1},
            {"index": 1, "status": "succeeded", "contrast_score": .4}], "encoded_crop_count": 2}
        description = "The building between two roads, without a red roof"
        result = rank_whole_crop(scorer, None, description, [[0, 0, 3, 3], [4, 4, 7, 7]],
                                 [1., .98], 0, 0)
        self.assertEqual(scorer.score.call_args.args[2:], (description, []))
        self.assertFalse(result["parsed"])
        self.assertEqual(result["pad_full_index"], 1)
        self.assertEqual(result["pad_guarded_index"], 0)

    def test_ties_invalid_scores_and_incomplete_evidence_keep_native(self):
        equal = [{"index": i, "status": "succeeded", "contrast_score": .2} for i in range(2)]
        self.assertEqual(select_whole_crop(equal, [1, 1], 1), 1)
        self.assertEqual(select_whole_crop(equal, [float("nan"), 1], 1), 1)
        invalid = [dict(equal[0], status="invalid_candidate_area"), equal[1]]
        self.assertEqual(select_whole_crop(invalid, [1, 1], 1), 1)

    def test_optional_failure_preserves_guarded_baseline(self):
        scorer = Mock()
        scorer.score.side_effect = RuntimeError("private details")
        result = rank_whole_crop(scorer, None, "a ship", np.ones((2, 4)), [1, 1], 0, 1)
        self.assertEqual(result["pad_full_index"], 1)
        self.assertNotIn("private", str(result))

    def test_incomplete_crops_preserve_guarded_rather_than_native_baseline(self):
        scorer = Mock()
        scorer.score.return_value = {"records": [
            {"index": 0, "status": "invalid_candidate_area", "contrast_score": None},
            {"index": 1, "status": "succeeded", "contrast_score": .3}], "encoded_crop_count": 1}
        result = rank_whole_crop(scorer, None, "a ship", np.ones((2, 4)), [1, 1], 0, 1)
        self.assertEqual(result["pad_full_index"], 1)
        self.assertEqual(result["status"], "incomplete_matching_evidence")


if __name__ == "__main__":
    unittest.main()
