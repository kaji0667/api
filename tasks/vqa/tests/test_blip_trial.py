"""Meaningful checks for trial selection, isolation and complete-denominator scoring."""
import copy
import unittest

from tasks.vqa.blip_trial import score_predictions, select_balanced
from tasks.vqa.config import SUPPORTED


class PublicTrialProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(sample_id=f"{kind}:{i}", question_type=kind, answer="hidden")
                     for kind in SUPPORTED for i in range(4)]

    def test_selection_does_not_depend_on_answers_or_input_order(self):
        chosen = select_balanced(self.rows, 2, 20261013)
        changed = copy.deepcopy(self.rows[::-1])
        for row in changed:
            row["answer"] = "a completely different answer"
        self.assertEqual([r["sample_id"] for r in chosen],
                         [r["sample_id"] for r in select_balanced(changed, 2, 20261013)])
        self.assertEqual(len(chosen), len(SUPPORTED)*2)

    def test_failures_count_wrong_and_paired_gain_loss_are_separate(self):
        rows = select_balanced(self.rows, 1, 20261013)
        gold = [dict(sample_id=r["sample_id"], question_type=r["question_type"], normalized_answer="gray") for r in rows]
        native = [dict(sample_id=r["sample_id"], result=dict(status="ok", answer_text="Grey."), seconds=1.) for r in rows]
        blip = copy.deepcopy(native)
        native[0]["result"] = dict(status="error", answer_text="gray")
        blip[1]["result"] = dict(status="ok", answer_text="red")
        report = score_predictions(rows, gold, dict(native=native, blip=blip))
        self.assertEqual(report["metrics"]["native"]["n"], 12)
        self.assertEqual(report["metrics"]["native"]["correct"], 11)
        self.assertEqual(report["metrics"]["native"]["failures"], 1)
        self.assertEqual(report["paired_blip_vs_native"], dict(gain=1, loss=1, both_correct=10, both_wrong=0))

    def test_incomplete_or_reordered_predictions_rejected(self):
        rows = select_balanced(self.rows, 1, 20261013)
        gold = [dict(sample_id=r["sample_id"], question_type=r["question_type"], normalized_answer="gray") for r in rows]
        predictions = [dict(sample_id=r["sample_id"], result=dict(status="ok", answer_text="gray"), seconds=1.) for r in rows]
        for invalid in (predictions[:-1], predictions[::-1]):
            with self.assertRaises(ValueError):
                score_predictions(rows, gold, dict(native=predictions, blip=invalid))


if __name__ == "__main__":
    unittest.main()
