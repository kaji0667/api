"""Independent detection geometry expectation retains the original evaluator output."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_parity import _detection_reference


class DetectionGeometryReferenceTests(unittest.TestCase):
    def test_reference_returns_semantic_scores_before_any_position_bonus(self):
        candidates = torch.tensor([[16., 24., 176., 54., .9, 0.],
                                   [656., 24., 816., 54., .2, 0.]])
        maps = [torch.tensor([[[[.48, .5]]]])]
        native = SimpleNamespace(
            device=torch.device("cpu"), backend=object(), spatial_weight=.15,
            encoder=SimpleNamespace(encoder=SimpleNamespace(
                encode=Mock(return_value=torch.ones((1, 768))),
            )),
        )
        with patch("tasks.detection.test._run_text_inference",
                   return_value=(torch.zeros((1, 5, 2)), maps, [], [])), \
                patch("ultralytics.utils.ops.non_max_suppression", return_value=[candidates]):
            result = _detection_reference(
                native, torch.zeros((1, 3, 128, 832)), "a target at the left",
                ((1., 1.), (16., 14.)), (100, 800), match=True,
            )
        torch.testing.assert_close(result["phrase_scores"], torch.tensor([[.48, .5]]), atol=0, rtol=0)
        self.assertEqual(result["selected"], 1)

    def test_reference_keeps_network_legacy_and_per_axis_boxes_separate(self):
        original = np.array([[1000., 20.2, 9000., 80.8]], dtype=np.float64)
        gains = np.array([.16, 17 / 101, .16, 17 / 101], dtype=np.float64)
        network = torch.from_numpy(original * gains + 16)
        candidates = torch.cat((network, torch.tensor([[.8, 0.]], dtype=torch.float64)), dim=1)
        raw = torch.zeros((1, 5, 1))
        native = SimpleNamespace(
            device=torch.device("cpu"), backend=object(),
            encoder=SimpleNamespace(encoder=SimpleNamespace(
                encode=Mock(return_value=torch.ones((1, 768))),
            )),
        )
        with patch("tasks.detection.test._run_text_inference", return_value=(raw, [], [], [])), \
                patch("ultralytics.utils.ops.non_max_suppression", return_value=[candidates]):
            result = _detection_reference(
                native, torch.zeros((1, 3, 64, 1632)), "a synthetic warehouse",
                ((17 / 101, .16), (16, 16)), (101, 10000), match=False,
            )
        np.testing.assert_allclose(result["geometry_boxes"], original, atol=1e-10, rtol=0)
        torch.testing.assert_close(result["network_boxes"], network, atol=0, rtol=0)
        legacy_expected = (network.numpy() - 16) / (17 / 101)
        np.testing.assert_allclose(result["boxes"].numpy(), legacy_expected, atol=1e-10, rtol=0)
        self.assertGreater(float(np.max(np.abs(result["geometry_boxes"] - result["boxes"].numpy()))), 400)


if __name__ == "__main__":
    unittest.main()
