"""CPU fixtures exercise original tensor inference without Web Predictor calls."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR.parent))
sys.path.insert(0, str(API_DIR))

from api_task_text import TaskInputError
import native_classification
import native_counting


_ORIGINAL_THREADS = None


def setUpModule():
    global _ORIGINAL_THREADS
    _ORIGINAL_THREADS = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    if _ORIGINAL_THREADS is not None:
        torch.set_num_threads(_ORIGINAL_THREADS)


class _SceneNetwork(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.head_loads = []
        self.inputs = []

    def load_head_checkpoint(self, checkpoint, strict=True):
        self.head_loads.append((checkpoint, strict))

    def forward(self, images):
        self.inputs.append(images.detach().clone())
        return self.logits


class NativeClassificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = [Path(self.tmp.name) / name for name in ("head.pt", "pretrain.pt", "model.yaml")]
        for path in self.paths:
            path.write_bytes(b"test fixture; never loaded")
        self.payload = {"classes": ["harbor", "airport", "bridge"], "num_classes": 3}
        self.network = _SceneNetwork(torch.tensor([[0.0, 3.0, 1.0]]))
        web_guard = patch("tasks.classification.inference.SceneClassificationPredictor", side_effect=AssertionError("Web Predictor used"))
        web_guard.start()
        self.addCleanup(web_guard.stop)

    def build(self, payload=None):
        with patch.object(native_classification.torch, "load", return_value=payload or self.payload), \
             patch.object(native_classification, "_create_network", return_value=self.network):
            return native_classification.NativeClassification(*self.paths, device="cpu")

    def test_strict_head_class_order_and_original_transform_softmax(self):
        runtime = self.build()
        self.assertEqual(self.network.head_loads, [(self.paths[0].resolve(), True)])
        self.assertFalse(self.network.training)
        image = Image.new("RGB", (100, 50), (255, 0, 0))
        result = runtime.predict(image, topk=3)
        self.assertEqual(result["original_size"], [50, 100])
        self.assertEqual(result["imgsz"], 640)
        tensor = self.network.inputs[-1]
        self.assertEqual(tuple(tensor.shape), (1, 3, 640, 640))
        expected_pixel = torch.tensor([(1.0 - .485) / .229, -.456 / .224, -.406 / .225])
        torch.testing.assert_close(tensor[0, :, 100, 100], expected_pixel)
        self.assertEqual([p["class_id"] for p in result["predictions"]], [1, 2, 0])
        self.assertEqual([p["class_name"] for p in result["predictions"]], ["airport", "bridge", "harbor"])
        self.assertAlmostEqual(sum(p["probability"] for p in result["predictions"]), 1.0, places=6)

    def test_class_count_mismatch_fails_before_network_construction(self):
        with patch.object(native_classification.torch, "load", return_value={"classes": ["harbor"], "num_classes": 2}), \
             patch.object(native_classification, "_create_network") as create:
            with self.assertRaisesRegex(ValueError, "class count"):
                native_classification.NativeClassification(*self.paths, device="cpu")
            create.assert_not_called()

    def test_scaled_yaml_alias_keeps_original_m_scale_without_alias_file(self):
        from ultralytics.nn.tasks import yaml_model_load

        alias = API_DIR.parent / "ultralytics/cfg/models/v12/yolov12m-classification.yaml"
        self.assertFalse(alias.is_file())
        self.assertEqual(yaml_model_load(str(alias))["scale"], "m")
        with patch.object(native_classification.torch, "load", return_value=self.payload), \
             patch.object(native_classification, "_create_network", return_value=self.network) as create:
            native_classification.NativeClassification(self.paths[0], self.paths[1], alias, device="cpu")
        create.assert_called_once_with(str(alias.resolve()), 3, str(self.paths[1].resolve()), "cpu")

    def test_invalid_logits_do_not_generate_an_answer(self):
        runtime = self.build()
        for logits in (torch.zeros(1, 2), torch.tensor([[float("nan"), 1., 2.]])):
            self.network.logits = logits
            with self.subTest(shape=tuple(logits.shape)), self.assertRaises(ValueError):
                runtime.predict(Image.new("RGB", (8, 8)), topk=3)

    def test_topk_must_be_positive(self):
        runtime = self.build()
        with self.assertRaisesRegex(ValueError, "topk"):
            runtime.predict(Image.new("RGB", (8, 8)), topk=0)


class _SharedEncoder:
    def __init__(self):
        self.queries = []

    def encode_counting(self, target):
        self.queries.append(target)
        return torch.ones(1, 768) / (768 ** .5)


class _Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Two overlapping boxes and one below threshold, in letterbox pixels.
        self.raw = torch.tensor([[
            [400., 401., 100.], [400., 401., 100.],
            [80., 80., 40.], [80., 80., 40.], [.9, .8, .1],
        ]])
        self.calls = []

    def forward(self, images, txt_vec, txt_token_mask):
        self.calls.append((images.detach().clone(), txt_vec.detach().clone(), txt_token_mask.detach().clone()))
        return self.raw, None


class NativeCountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkpoint = Path(self.tmp.name) / "count.pt"
        self.checkpoint.write_bytes(b"test fixture; never loaded")
        self.encoder = _SharedEncoder()
        self.model = _Detector()
        self.image = Image.new("RGB", (1600, 800), (20, 40, 60))
        for target in ("tasks.counting.inference.CountingPredictor", "tasks.detection.inference.TextGuidedDetectionPredictor"):
            web_guard = patch(target, side_effect=AssertionError("Web Predictor used"))
            web_guard.start()
            self.addCleanup(web_guard.stop)

    def build(self):
        with patch.object(native_counting, "_load_model", return_value=self.model):
            return native_counting.NativeCounting(self.checkpoint, device="cpu", encoder=self.encoder)

    def test_simple_question_reaches_original_counter_with_nms_and_rgb_letterbox(self):
        runtime = self.build()
        result = runtime.predict(self.image, "How many ships are in the image?\nOnly answer a number.")
        self.assertEqual(result.target_class, "ship")
        self.assertEqual(self.encoder.queries, ["ship"])
        self.assertEqual(result.count, 1)  # Actual original class-agnostic NMS suppresses duplicate.
        self.assertEqual(result.original_size, (800, 1600))
        np.testing.assert_allclose(result.boxes, [[720., 320., 880., 480.]], atol=1e-5)
        np.testing.assert_allclose(result.scores, [.9], atol=1e-6)
        images, text, mask = self.model.calls[-1]
        self.assertEqual(tuple(images.shape), (1, 3, 800, 800))
        torch.testing.assert_close(images[0, :, 400, 400], torch.tensor([20., 40., 60.]) / 255.)
        torch.testing.assert_close(images[0, :, 10, 10], torch.full((3,), 114. / 255.))
        self.assertEqual(tuple(text.shape), (1, 1, 768))
        self.assertEqual(text.device, runtime.device)
        self.assertEqual(tuple(mask.shape), (1, 1))
        self.assertEqual(mask.dtype, torch.bool)

    def test_empty_detections_remain_zero_count(self):
        runtime = self.build()
        self.model.raw[:, 4, :] = .1
        result = runtime.predict(self.image, "ship")
        self.assertEqual(result.count, 0)
        self.assertEqual(result.boxes.shape, (0, 4))
        self.assertEqual(result.scores.shape, (0,))

    def test_qualified_count_passes_extracted_class_to_original_counter(self):
        runtime = self.build()
        raw = self.model.raw.clone()
        plain = runtime.predict(self.image, "How many ships are in the image?")
        self.model.raw = raw.clone()
        qualified = runtime.predict(self.image, "How many ships are in the upper-left corner?")
        self.assertEqual(self.encoder.queries, ["ship", "ship"])
        self.assertEqual((plain.count, qualified.count), (1, 1))
        # The region neither crops the image nor filters the model's detections.
        torch.testing.assert_close(self.model.calls[0][0], self.model.calls[1][0], rtol=0, atol=0)
        np.testing.assert_array_equal(plain.boxes, qualified.boxes)

    def test_plane_synonym_reaches_counter_as_original_class(self):
        runtime = self.build()
        runtime.predict(self.image, "How many planes are in the left half?")
        self.assertEqual(self.encoder.queries, ["airplane"])
        self.assertEqual(len(self.model.calls), 1)

    def test_encoder_failure_prevents_forward(self):
        runtime = self.build()
        with patch.object(self.encoder, "encode_counting", side_effect=TaskInputError("text_context_exceeded", "too long")):
            with self.assertRaises(TaskInputError):
                runtime.predict(self.image, "ship")
        self.assertFalse(self.model.calls)

    def test_mismatched_count_rejected(self):
        runtime = self.build()
        with patch.object(runtime.counter, "count", return_value={
            "pred_count": 3, "pred_boxes": torch.empty(0, 4), "pred_scores": torch.empty(0),
        }):
            with self.assertRaisesRegex(ValueError, "NMS detection count"):
                runtime.predict(self.image, "ship")


if __name__ == "__main__":
    unittest.main()
