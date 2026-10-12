"""Threshold trial structural gates; no model assets or public GT are loaded."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from try_crop_candidate_thresholds import (ARMS, COHORTS, DEADLINE_UTC, PROTOCOL, THRESHOLDS,
                                          add_guarded_arm, score_run, validate_prediction_rows)
from try_crop_identity import sha256


class ThresholdTrialTests(unittest.TestCase):
    def fixture(self):
        memberships = [{"cohort": "public115", "id": f"val_{i}", "row_key": f"public115:val_{i}"}
                       for i in range(115)]
        memberships += [{"cohort": "public80", "id": f"val_{i}", "row_key": f"public80:val_{i}"}
                        for i in [*range(5), *range(115, 190)]]
        frozen = {"protocol": PROTOCOL, "thresholds": list(THRESHOLDS), "arms": list(ARMS),
                  "membership_count": 195, "unique_id_count": 190, "memberships": memberships,
                  "deadline_utc": DEADLINE_UTC, "annotations_sha256": "unused_without_full_prediction",
                  "registered_utc": "2026-10-12T19:00:00Z",
                  "manifest_sha256": {cohort: spec["sha256"] for cohort, spec in COHORTS.items()}}
        completion = {"protocol": PROTOCOL, "completed": True, "prediction_count": 585,
                      "membership_count": 195, "unique_id_count": 190,
                      "completed_utc": "2026-10-12T21:00:00Z"}
        rows = []
        for member in memberships:
            for threshold in THRESHOLDS:
                row = {**member, "candidate_confidence": threshold, "candidate_boxes": [[0., 0., 10., 10.], [20., 20., 30., 30.]],
                       "completed_utc": "2026-10-12T20:00:00Z", "phrase_scores": [.5, .51],
                       "native_confidence_scores": [.8, .7], "native_status": "succeeded",
                       "native_diagnostics": {"confidence_threshold": threshold, "nms_candidate_count": 2},
                       "parse_status": "parsed", "arms": {
                           "A": {"selected_index": 0, "selected_box_xyxy": [0., 0., 10., 10.], "status": "succeeded"},
                           "E": {"selected_index": 1, "selected_box_xyxy": [20., 20., 30., 30.], "status": "succeeded"}}}
                add_guarded_arm(row)
                rows.append(row)
        return rows, frozen, completion

    def test_full_memberships_keep_overlap_in_two_cohorts(self):
        rows, frozen, completion = self.fixture()
        validate_prediction_rows(rows, frozen, completion)
        self.assertEqual(sum(row["cohort"] == "public115" for row in rows), 345)
        self.assertEqual(sum(row["cohort"] == "public80" for row in rows), 240)

    def test_every_threshold_required_and_duplicates_cannot_replace_missing_rows(self):
        rows, frozen, completion = self.fixture()
        for invalid in (rows[:-1], rows[:195], [*rows[:-1], rows[0]]):
            with self.subTest(size=len(invalid)), self.assertRaises(ValueError):
                validate_prediction_rows(invalid, frozen, completion)

    def test_g_only_follows_parse_capability_and_keeps_original_e(self):
        rows, _, _ = self.fixture()
        row = rows[0]
        original_e = copy.deepcopy(row["arms"]["E"])
        self.assertEqual(row["arms"]["G"]["selected_index"], 1)
        row["parse_status"] = "complete_description_fallback"
        row["id"] = "an_arbitrary_id_without_metric"
        add_guarded_arm(row)
        self.assertEqual(row["arms"]["G"]["selected_index"], 0)
        self.assertTrue(row["G_fallback_A"])
        self.assertEqual(row["arms"]["E"], original_e)
        row["parse_status"] = "parsed"
        row["arms"]["E"] = dict(row["arms"]["A"], status="original_selection_fallback")
        add_guarded_arm(row)
        self.assertEqual(row["arms"]["G"]["selected_index"], 0)
        self.assertEqual(row["arms"]["G"]["status"], "original_selection_fallback")
        self.assertTrue(row["G_used_E"])

    def test_failed_native_rows_keep_full_denominator(self):
        rows, frozen, completion = self.fixture()
        rows[0].update(native_status="native_prediction_failed", native_diagnostics=None,
                       candidate_boxes=[], phrase_scores=[], native_confidence_scores=[])
        for arm in ("A", "E"):
            rows[0]["arms"][arm] = {"selected_index": None, "selected_box_xyxy": None, "status": "no_original_selection"}
        add_guarded_arm(rows[0])
        validate_prediction_rows(rows, frozen, completion)
        self.assertEqual(len(rows), 585)

    def test_diagnostics_selection_and_g_cannot_drift_from_pool(self):
        rows, frozen, completion = self.fixture()
        mutations = (
            lambda row: row["native_diagnostics"].update(confidence_threshold=.20),
            lambda row: row["arms"]["G"].update(selected_index=0),
            lambda row: row["arms"]["E"].update(selected_box_xyxy=[0, 0, 1, 1]),
        )
        for mutate in mutations:
            invalid = copy.deepcopy(rows)
            mutate(invalid[0])
            with self.assertRaises(ValueError):
                validate_prediction_rows(invalid, frozen, completion)

    def test_interruption_or_deadline_cannot_be_complete(self):
        rows, frozen, completion = self.fixture()
        for update in ({"completed": False}, {"completed_utc": "2026-10-12T23:00:01Z"}, {"prediction_count": 584}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_prediction_rows(rows, frozen, {**completion, **update})

    def test_fixed_manifest_counts_timeline_and_inverse_boolean_are_checked(self):
        rows, frozen, completion = self.fixture()
        bad_frozen = copy.deepcopy(frozen)
        bad_frozen["manifest_sha256"]["public80"] = "unregistered"
        with self.assertRaises(ValueError):
            validate_prediction_rows(rows, bad_frozen, completion)
        with self.assertRaises(ValueError):
            validate_prediction_rows(rows, frozen, {**completion, "membership_count": 190})
        invalid = copy.deepcopy(rows)
        invalid[0]["completed_utc"] = "2026-10-12T18:00:00Z"
        with self.assertRaises(ValueError):
            validate_prediction_rows(invalid, frozen, completion)
        invalid = copy.deepcopy(rows)
        invalid[0]["G_fallback_A"] = None
        with self.assertRaises(ValueError):
            validate_prediction_rows(invalid, frozen, completion)

    def test_score_refuses_partial_round_before_any_gold_file_access(self):
        rows, frozen, completion = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            prediction_file, frozen_file = directory / "predictions.jsonl", directory / "frozen_run.json"
            prediction_file.write_text("\n".join(json.dumps(row) for row in rows[:-1]), encoding="utf-8")
            frozen_file.write_text(json.dumps(frozen), encoding="utf-8")
            completion.update(predictions_sha256=sha256(prediction_file), frozen_run_sha256=sha256(frozen_file))
            (directory / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
            gold_path = directory / "do_not_open_gold.jsonl"
            with patch("try_crop_candidate_thresholds._public_rows") as read_gold:
                with self.assertRaisesRegex(ValueError, "memberships are required"):
                    score_run(directory, gold_path)
                read_gold.assert_not_called()


if __name__ == "__main__":
    unittest.main()
