from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from tasks.detection.crop_identity import CropIdentityError, FrozenCLIPCropScorer, rank_full_candidates
from scripts.try_crop_identity import (PROTOCOL, prepare80, score_run, sha256, without_gold, write_new_json)


CONFIG = {"size": 224, "mode": "RGB", "mean": (0.48145466, 0.4578275, 0.40821073),
          "std": (0.26862954, 0.26130258, 0.27577711), "interpolation": "bicubic",
          "resize_mode": "shortest", "fill_color": 0}


class DummyCLIP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.visual = SimpleNamespace(preprocess_cfg=dict(CONFIG))
        self.encode_calls = 0
        self.encoded_images = 0
        self.eval()

    def encode_image(self, images, normalize=False):
        self.encode_calls += 1
        self.encoded_images += len(images)
        red = images[:, 0].mean((1, 2)) >= images[:, 2].mean((1, 2))
        return torch.stack([torch.tensor([.8, .6]) if flag else torch.tensor([.7, -(1 - .7 ** 2) ** .5])
                            for flag in red])


class CropIdentityTests(unittest.TestCase):
    def setUp(self):
        self.model = DummyCLIP()
        self.pixels = []
        self.configurations = []

        def transform_factory(configuration):
            self.configurations.append(configuration)

            def transform(image):
                self.pixels.append(np.asarray(image).copy())
                # Fixed-size test transform, preserving RGB channel means.
                average = np.asarray(image).mean((0, 1))
                return torch.tensor(average).float().view(3, 1, 1).expand(3, 2, 2)
            return transform

        def encode(texts):
            return torch.stack([torch.tensor([.8, .6]) if "road" in text else torch.tensor([1., 0.]) for text in texts])

        self.scorer = FrozenCLIPCropScorer(self.model, encode, batch_size=1, transform_factory=transform_factory)
        self.image = np.zeros((4, 8, 3), dtype=np.uint8)
        self.image[:, :4, 2] = 255  # BGR red -> RGB red.
        self.image[:, 4:, 0] = 255  # BGR blue -> RGB blue.
        self.boxes = [[0., 0., 4., 4.], [4., 0., 8., 4.]]

    def test_identity_contrast_three_arms_and_each_crop_encoded_once(self):
        result = rank_full_candidates(self.scorer, self.image, "bridge above road", self.boxes, 1,
                                      phrase_scores=[.1, .9])
        self.assertEqual(result["arms"]["A"]["selected_index"], 1)
        self.assertEqual(result["arms"]["B"]["selected_index"], 0)
        self.assertEqual(result["arms"]["C"]["selected_index"], 1)
        self.assertEqual(self.model.encoded_images, 2)
        self.assertEqual(result["image_encode_batch_count"], 2)
        self.assertEqual(result["candidate_boxes"], self.boxes)
        self.assertEqual(result["phrase_scores"], [.1, .9])
        self.assertEqual(result["parse_status"], "parsed")
        np.testing.assert_array_equal(self.pixels[0][0, 0], [255, 0, 0])
        self.assertEqual(self.configurations[0], CONFIG)
        without_gold(result)

    def test_ties_prefer_original_then_lowest_index(self):
        identical = [self.boxes[0], self.boxes[0]]
        result = rank_full_candidates(self.scorer, self.image, "bridge", identical, 1)
        self.assertEqual(result["arms"]["B"]["selected_index"], 1)
        self.assertEqual(result["arms"]["C"]["selected_index"], 1)
        result = rank_full_candidates(self.scorer, self.image, "bridge", identical, None)
        self.assertEqual(result["arms"]["B"]["selected_index"], 0)

    def test_crop_clips_then_rounds_outward(self):
        result = self.scorer.score(self.image, [[-1.2, -.1, 3.2, 3.1]], "bridge", [])
        record = result["records"][0]
        self.assertEqual(record["crop_bounds_xyxy"], [0, 0, 4, 4])
        self.assertEqual(record["crop_aspect_ratio"], 1.)

    def test_empty_and_zero_area_candidates_are_explicit(self):
        result = rank_full_candidates(self.scorer, self.image, "bridge", [], None)
        self.assertEqual(result["score_status"], "empty_candidates")
        self.assertEqual(self.model.encoded_images, 0)
        result = rank_full_candidates(self.scorer, self.image, "bridge", [[1, 1, 1, 2]], 0)
        self.assertEqual(result["arms"]["A"]["selected_index"], 0)
        self.assertEqual(result["arms"]["B"]["status"], "no_valid_crop")
        self.assertEqual(result["crop_scores"][0]["status"], "invalid_candidate_area")

    def test_bad_features_do_not_fallback_to_phrase_scores(self):
        self.scorer.encode_descriptions = lambda texts: torch.full((len(texts), 2), float("nan"))
        result = rank_full_candidates(self.scorer, self.image, "bridge", self.boxes, 1, phrase_scores=[.1, .9])
        self.assertEqual(result["score_status"], "invalid_clip_text_features")
        self.assertEqual(result["arms"]["A"]["selected_index"], 1)
        self.assertIsNone(result["arms"]["B"]["selected_index"])
        self.assertIsNone(result["arms"]["C"]["selected_index"])

    def test_unfrozen_model_and_missing_transform_fail_explicitly(self):
        self.model.anchor.requires_grad_(True)
        with self.assertRaisesRegex(CropIdentityError, "clip_not_frozen"):
            FrozenCLIPCropScorer(self.model, self.scorer.encode_descriptions)
        self.model.anchor.requires_grad_(False)
        self.model.visual.preprocess_cfg = {}
        with self.assertRaisesRegex(CropIdentityError, "clip_transform_unavailable"):
            FrozenCLIPCropScorer(self.model, self.scorer.encode_descriptions)

    def test_nonfinite_boxes_and_invalid_baseline_rejected(self):
        with self.assertRaisesRegex(CropIdentityError, "invalid_candidate_boxes"):
            rank_full_candidates(self.scorer, self.image, "bridge", [[0, 0, float("nan"), 3]], None)
        with self.assertRaisesRegex(CropIdentityError, "invalid_baseline_index"):
            rank_full_candidates(self.scorer, self.image, "bridge", self.boxes, True)
        with self.assertRaisesRegex(CropIdentityError, "invalid_candidate_boxes"):
            self.scorer.score(self.image, [[0, 0], [3, 3]], "bridge", [])

    def test_soft_geometry_is_separate_and_never_changes_candidates(self):
        anchors = [{"description": "road", "relation": "above", "selected_box": [1, 2, 3, 4]}]
        result = rank_full_candidates(self.scorer, self.image, "bridge above road", self.boxes, 0,
                                      geometry=True, anchors=anchors)
        self.assertEqual(result["candidate_boxes"], self.boxes)
        self.assertEqual(set(result["arms"]), {"A", "B", "C", "D"})
        for record in result["crop_scores"]:
            self.assertAlmostEqual(record["geometry_score"], record["contrast_score"] + .15 * record["geometry"])
        with self.assertRaisesRegex(CropIdentityError, "invalid_reference_anchors"):
            rank_full_candidates(self.scorer, self.image, "bridge above road", self.boxes, 0, geometry=True)

    def test_hybrid_preserves_raw_native_scale_and_fixed_correction(self):
        result = rank_full_candidates(self.scorer, self.image, "bridge above road", self.boxes, 0,
                                      phrase_scores=[.6, .1], hybrid=True, baseline_scores=[.6, .1])
        self.assertEqual(result["arms"]["C"]["selected_index"], 1)
        self.assertEqual(result["arms"]["E"]["selected_index"], 0)
        self.assertAlmostEqual(result["crop_scores"][0]["hybrid_score"], .6)
        self.assertAlmostEqual(result["crop_scores"][1]["hybrid_score"], .25)
        # A caller can inject the exact native absolute-position-adjusted scores.
        result = rank_full_candidates(self.scorer, self.image, "bridge above road", self.boxes, 1,
                                      phrase_scores=[.4, .4], hybrid=True, baseline_scores=[.4, .55])
        self.assertEqual(result["baseline_adjusted_scores"], [.4, .55])
        self.assertAlmostEqual(result["crop_scores"][1]["hybrid_score"], .7)

    def test_hybrid_failures_explicitly_keep_a_and_minmax_ties_are_zero(self):
        identical = [self.boxes[0], self.boxes[0]]
        result = rank_full_candidates(self.scorer, self.image, "bridge", identical, 1,
                                      phrase_scores=[.2, .2], hybrid=True, baseline_scores=[.2, .2])
        self.assertEqual(result["arms"]["E"]["selected_index"], 1)
        self.assertEqual([record["contrast_pool_minmax"] for record in result["crop_scores"]], [0., 0.])
        self.scorer.encode_descriptions = lambda texts: torch.full((len(texts), 2), float("nan"))
        result = rank_full_candidates(self.scorer, self.image, "bridge", self.boxes, 1,
                                      phrase_scores=[.1, .9], hybrid=True, baseline_scores=[.1, .9])
        self.assertEqual(result["arms"]["E"]["status"], "original_selection_fallback")
        self.assertEqual(result["arms"]["E"]["selected_index"], 1)
        self.assertEqual(result["arms"]["E"]["fallback_reason"], "invalid_clip_text_features")

    def test_gold_schema_rejected_recursively(self):
        with self.assertRaisesRegex(ValueError, "Gold"):
            without_gold({"samples": [{"candidate_boxes": [], "segmentation": [1]}]})

    def test_score_requires_completed_unchanged_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "val.jsonl"
            annotation.write_text(json.dumps({"id": "val_1", "split": "val", "bbox": [0, 0, 4, 4]}) + "\n")
            prediction = rank_full_candidates(self.scorer, self.image, "bridge", self.boxes, 0)
            prediction.update(id="val_1", image_hw=[4, 8])
            path = root / "predictions.jsonl"
            path.write_text(json.dumps(prediction) + "\n")
            frozen_path = root / "frozen_run.json"
            write_new_json(frozen_path, {"annotations_sha256": sha256(annotation), "arms": ["A", "B", "C"]})
            with self.assertRaises(FileNotFoundError):
                score_run(root, annotation)
            write_new_json(root / "completion.json", {"completed": True, "protocol": PROTOCOL, "sample_count": 1,
                "predictions_sha256": sha256(path), "frozen_run_sha256": sha256(frozen_path)})
            result = score_run(root, annotation)
            self.assertEqual(result["summary"]["A"]["correct_at_0_5"], 1)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "complete"):
                score_run(root, annotation)

    def test_public80_selection_uses_class4_seed42_without_gold_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for category in range(20):
                for index in range(5):
                    name = f"{category}_{index}.jpg"
                    (root / name).touch()
                    rows.append({"id": f"val_{category}_{index}", "split": "val", "class_idx": category,
                                 "image": name, "text": "bridge", "bbox": [0, 0, 1, 1], "segmentation": "unused"})
            annotations = root / "val.jsonl"
            annotations.write_text("\n".join(json.dumps(row) for row in rows))
            first = prepare80(annotations, root)
            for row in rows:
                row["bbox"] = [100, 100, 200, 200]
                row["segmentation"] = "changed gold"
            annotations.write_text("\n".join(json.dumps(row) for row in rows))
            second = prepare80(annotations, root)
            self.assertEqual(first["samples"], second["samples"])
            self.assertEqual(len(first["samples"]), 80)
            without_gold(second)


if __name__ == "__main__":
    unittest.main()
