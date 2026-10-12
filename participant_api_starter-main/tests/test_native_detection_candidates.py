"""Opt-in same-forward phrase scoring; synthetic maps and mocked model loading."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_task_text import TaskInputError
from native_detection import NativeDetection
from text_encoder.matching import score_matrix_from_phrase_maps


class NativeDetectionCandidateScoringTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(return_value=torch.zeros((1, 5, 2)))
        self.model._last_text_outputs = {"phrase_logits": [torch.tensor([[[[1., 0.], [0., 5.]]]])]}
        backend = Mock(pt=True, nn_module=False, model=self.model, stride=32)
        for boundary in (
            patch("native_detection.AutoBackend", return_value=backend),
            patch("native_detection.Path.is_file", return_value=True),
            patch("native_detection.detection_text", return_value="The striped courtyard"),
        ):
            boundary.start()
            self.addCleanup(boundary.stop)
        self.detections = torch.tensor([
            [16., 16., 416., 416., .95, 0.],
            [416., 416., 816., 816., .30, 0.],
        ])
        nms_patch = patch("native_detection.non_max_suppression", return_value=[self.detections])
        self.nms = nms_patch.start()
        self.addCleanup(nms_patch.stop)
        encoder = Mock()
        encoder.encode_descriptions.return_value = torch.ones((1, 768)) / math.sqrt(768)
        self.native = NativeDetection("synthetic.pt", device="cpu", encoder=encoder, spatial_weight=0)
        self.image = Image.new("RGB", (80, 80), (11, 22, 33))
        self.question = "Synthetic public description with output packaging."

    def test_returned_phrase_scores_distinguish_identity_from_detection_confidence(self):
        result = self.native.predict(self.image, self.question, return_phrase_scores=True)
        np.testing.assert_allclose(result.scores, [.95, .30])
        np.testing.assert_array_equal(result.phrase_scores, [1., 5.])
        self.assertEqual(result.phrase_scores.shape, (2,))
        self.assertEqual(result.phrase_scores.dtype, np.float32)
        self.assertEqual(result.grounding_index, 1)
        self.assertEqual(result.detection_count, 2)
        self.assertIsNone(result.extra_phrase_scores)
        self.model.assert_called_once()

    def test_scores_are_returned_before_absolute_position_bonus(self):
        self.native.spatial_weight = .15
        raw_phrase_scores = torch.tensor([[1.02, 1.]])
        with patch("native_detection.detection_text", return_value="The courtyard in the bottom-right corner"), \
                patch("native_detection.score_matrix_from_phrase_maps", return_value=raw_phrase_scores):
            result = self.native.predict(self.image, self.question, return_phrase_scores=True)
        np.testing.assert_array_equal(result.phrase_scores, raw_phrase_scores[0].numpy())
        self.assertEqual(result.grounding_index, 1)
        torch.testing.assert_close(raw_phrase_scores, torch.tensor([[1.02, 1.]]), atol=0, rtol=0)

    def test_external_and_nms_boxes_use_one_forward_and_one_matching_call(self):
        external = np.array([[0., 0., 80., 80.], [40., 40., 80., 80.]], dtype=np.float64)
        saved = external.copy()
        with patch("native_detection.score_matrix_from_phrase_maps", wraps=score_matrix_from_phrase_maps) as score:
            result = self.native.predict(self.image, self.question, return_phrase_scores=True,
                                         score_boxes_xyxy=external)
        self.model.assert_called_once()
        self.nms.assert_called_once()
        score.assert_called_once()
        torch.testing.assert_close(score.call_args.kwargs["candidate_boxes_xyxy"], torch.tensor([
            [16., 16., 416., 416.], [416., 416., 816., 816.],
            [16., 16., 816., 816.], [416., 416., 816., 816.],
        ]), atol=0, rtol=0)
        np.testing.assert_array_equal(result.phrase_scores, [1., 5.])
        np.testing.assert_array_equal(result.extra_phrase_scores, [1.5, 5.])
        np.testing.assert_array_equal(external, saved)
        self.assertEqual(result.extra_phrase_scores.dtype, np.float32)
        self.assertEqual(result.extra_phrase_scores.shape, (2,))
        self.assertEqual(result.detection_count, 2)

    def test_external_high_score_never_enters_native_candidate_assignment(self):
        with patch("native_detection.score_matrix_from_phrase_maps", return_value=torch.tensor([[1., 5., 100.]])):
            result = self.native.predict(self.image, self.question, return_phrase_scores=True,
                                         score_boxes_xyxy=np.array([[0., 0., 80., 80.]]))
        self.assertEqual(result.grounding_index, 1)
        self.assertEqual(result.detection_count, 2)
        np.testing.assert_array_equal(result.phrase_scores, [1., 5.])
        np.testing.assert_array_equal(result.extra_phrase_scores, [100.])
        np.testing.assert_array_equal(result.boxes, [[0., 0., 40., 40.], [40., 40., 80., 80.]])

    def test_panorama_scoring_uses_actual_per_axis_gains_and_padding(self):
        # Known ceil resize is 101x10000 -> 17x1600. Uniform .16 y gain is wrong.
        external = np.array([[1000., 20.2, 9000., 80.8]])
        self.nms.return_value = [torch.tensor([[176., 19.4, 1456., 29.6, .8, 0.]])]
        with patch.object(self.native, "_preprocess", return_value=(
                torch.zeros((1, 3, 64, 1632)), ((17 / 101, .16), (16, 16)))), \
                patch("native_detection.score_matrix_from_phrase_maps", wraps=score_matrix_from_phrase_maps) as score:
            result = self.native.predict(Image.new("RGB", (10000, 101)), self.question,
                                         return_phrase_scores=True, score_boxes_xyxy=external)
        self.assertEqual(score.call_args.kwargs["input_hw"], (64, 1632))
        torch.testing.assert_close(score.call_args.kwargs["candidate_boxes_xyxy"],
                                   torch.tensor([[176., 19.4, 1456., 29.6]]).repeat(2, 1), atol=2e-6, rtol=0)
        np.testing.assert_array_equal(result.phrase_scores, result.extra_phrase_scores)
        np.testing.assert_allclose(result.boxes, external, atol=.001, rtol=0)
        self.model.assert_called_once()

    def test_noninteger_resize_scores_external_and_local_identical_boxes_equally(self):
        # Original dataset preprocessing: 953x1047 -> 729x800, left=16/top=19.
        external = np.array([[101., 203., 941., 761.]], dtype=np.float64)
        expected = torch.tensor([[101 * 800 / 1047 + 16, 203 * 729 / 953 + 19,
                                  941 * 800 / 1047 + 16, 761 * 729 / 953 + 19]], dtype=torch.float32)
        self.nms.return_value = [torch.cat((expected, torch.tensor([[.8, 0.]])), dim=1)]
        with patch("native_detection.score_matrix_from_phrase_maps", wraps=score_matrix_from_phrase_maps) as score:
            result = self.native.predict(Image.new("RGB", (1047, 953)), self.question,
                                         return_phrase_scores=True, score_boxes_xyxy=external)
        torch.testing.assert_close(score.call_args.kwargs["candidate_boxes_xyxy"], expected.repeat(2, 1),
                                   atol=0, rtol=0)
        self.assertEqual(score.call_args.kwargs["input_hw"], (768, 832))
        np.testing.assert_array_equal(result.phrase_scores, result.extra_phrase_scores)

    def test_external_boxes_still_score_when_nms_is_empty(self):
        self.nms.return_value = [torch.empty((0, 6))]
        with patch("native_detection.score_matrix_from_phrase_maps", wraps=score_matrix_from_phrase_maps) as score:
            result = self.native.predict(self.image, self.question, return_phrase_scores=True,
                                         score_boxes_xyxy=[[0., 0., 40., 40.], [40., 40., 80., 80.]])
        self.assertEqual(result.boxes.shape, (0, 4))
        self.assertEqual(result.phrase_scores.shape, (0,))
        np.testing.assert_array_equal(result.extra_phrase_scores, [1., 5.])
        self.assertIsNone(result.grounding_index)
        self.model.assert_called_once()
        score.assert_called_once()

    def test_external_only_scoring_does_not_require_grounding_or_return_local_scores(self):
        with patch("native_detection.parse_absolute_position") as position, \
                patch("native_detection.assign_by_score_matrix") as assign:
            result = self.native.predict(self.image, self.question, require_grounding=False,
                                         score_boxes_xyxy=torch.tensor([[40., 40., 80., 80.]]))
        position.assert_not_called()
        assign.assert_not_called()
        self.assertIsNone(result.grounding_index)
        self.assertIsNone(result.phrase_scores)
        np.testing.assert_array_equal(result.extra_phrase_scores, [5.])

    def test_opt_in_empty_scores_do_not_require_phrase_maps_when_no_boxes_exist(self):
        self.nms.return_value = [torch.empty((0, 6))]
        self.model._last_text_outputs = {}
        with patch("native_detection.score_matrix_from_phrase_maps") as score:
            result = self.native.predict(self.image, self.question, return_phrase_scores=True,
                                         score_boxes_xyxy=np.empty((0, 4), dtype=np.float32))
        score.assert_not_called()
        self.assertEqual(result.phrase_scores.shape, (0,))
        self.assertEqual(result.extra_phrase_scores.shape, (0,))
        self.assertIsNone(result.grounding_index)
        self.nms.return_value = [self.detections]
        with patch("native_detection.score_matrix_from_phrase_maps") as score:
            presence = self.native.predict(self.image, self.question, require_grounding=False,
                                           score_boxes_xyxy=np.empty((0, 4), dtype=np.float32))
        score.assert_not_called()
        self.assertIsNone(presence.phrase_scores)
        self.assertEqual(presence.extra_phrase_scores.shape, (0,))

    def test_default_and_opt_in_results_have_exact_original_selection_parity(self):
        default = self.native.predict(self.image, self.question)
        enabled = self.native.predict(self.image, self.question, return_phrase_scores=True,
                                      score_boxes_xyxy=np.array([[0., 0., 80., 80.]]))
        self.assertIsNone(default.phrase_scores)
        self.assertIsNone(default.extra_phrase_scores)
        self.assertEqual(default.grounding_index, enabled.grounding_index)
        self.assertEqual(default.original_size, enabled.original_size)
        self.assertEqual(default.model_prompt, enabled.model_prompt)
        np.testing.assert_array_equal(default.boxes, enabled.boxes)
        np.testing.assert_array_equal(default.scores, enabled.scores)
        self.assertEqual(self.nms.call_args.kwargs, {
            "conf_thres": .25, "iou_thres": .7, "multi_label": False, "agnostic": True, "max_det": 300,
        })

    def test_default_presence_never_requests_phrase_maps(self):
        self.model._last_text_outputs = {}
        with patch("native_detection.score_matrix_from_phrase_maps") as score:
            result = self.native.predict(self.image, self.question, require_grounding=False)
        score.assert_not_called()
        self.assertIsNone(result.phrase_scores)
        self.assertIsNone(result.extra_phrase_scores)
        self.assertIsNone(result.grounding_index)

    def test_returned_scores_are_arrays_without_map_or_score_tensor_aliases(self):
        raw = torch.tensor([[1., 5., 8.]])
        with patch("native_detection.score_matrix_from_phrase_maps", return_value=raw):
            result = self.native.predict(self.image, self.question, return_phrase_scores=True,
                                         score_boxes_xyxy=np.array([[0., 0., 80., 80.]]))
        result.phrase_scores[0] = -9
        result.extra_phrase_scores[0] = -10
        torch.testing.assert_close(raw, torch.tensor([[1., 5., 8.]]), atol=0, rtol=0)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in vars(result).values()))

    def test_invalid_external_boxes_are_rejected_in_original_coordinates(self):
        cases = (
            np.array([0., 0., 80., 80.]), np.ones((1, 3)), np.empty((0, 3)),
            np.array([[0., 0., float("nan"), 40.]]), np.array([[0., 0., 40., float("inf")]]),
            np.array([[-1., 0., 40., 40.]]), np.array([[0., 0., 81., 40.]]),
            np.array([[0., 0., 40., 81.]]), np.array([[10., 10., 10., 20.]]),
            np.array([[10., 10., 20., 10.]]), np.array([[20., 10., 10., 20.]]),
            np.array([[0, 0, 1, 1]], dtype=bool), np.array([["0", "0", "1", "1"]]),
            np.ones((1, 4), dtype=complex), np.ones((1, 4), dtype=object),
            [[0., 0., 20., 20.], [0., 0.]],
        )
        for boxes in cases:
            with self.subTest(boxes=repr(boxes)), self.assertRaises(TaskInputError) as context:
                self.native._score_boxes_for_input(boxes, (80, 80), ((10, 10), (16, 16)))
            self.assertEqual(context.exception.code, "invalid_detection_geometry")

    def test_invalid_external_box_fails_before_forward(self):
        with self.assertRaises(TaskInputError) as context:
            self.native.predict(self.image, self.question, score_boxes_xyxy=[[0., 0., 81., 80.]])
        self.assertEqual(context.exception.code, "invalid_detection_geometry")
        self.model.assert_not_called()
        self.nms.assert_not_called()

    def test_invalid_scoring_metadata_and_flag_fail_explicitly(self):
        for ratio_pad in (((0, 1), (0, 0)), ((1, float("nan")), (0, 0)),
                          ((1, 1), (-1, 0)), ((1,), (0, 0)), ((1e308, 1e308), (0, 0))):
            with self.subTest(ratio_pad=ratio_pad), self.assertRaises(TaskInputError):
                self.native._score_boxes_for_input([[0., 0., 10., 10.]], (80, 80), ratio_pad)
        for flag in (1, None, "true", []):
            with self.subTest(flag=flag), self.assertRaises(TaskInputError) as context:
                self.native.predict(self.image, self.question, return_phrase_scores=flag)
            self.assertEqual(context.exception.code, "invalid_detection_matching")
        self.model.assert_not_called()

    def test_invalid_same_forward_phrase_matrix_cannot_be_returned(self):
        for values in (torch.tensor([[1., 5.]]), torch.tensor([[1., 5., float("nan")]])):
            with self.subTest(values=values), \
                    patch("native_detection.score_matrix_from_phrase_maps", return_value=values), \
                    self.assertRaises(TaskInputError) as context:
                self.native.predict(self.image, self.question, return_phrase_scores=True,
                                    score_boxes_xyxy=[[0., 0., 80., 80.]])
            self.assertEqual(context.exception.code, "invalid_detection_matching")


if __name__ == "__main__":
    unittest.main()
