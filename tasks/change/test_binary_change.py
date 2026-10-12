"""Synthetic input and strict loading boundaries for the independent model."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from .binary_change import (ARCHITECTURE_ID, NETWORK_SHA256, SOURCE_REVISION,
                            BinaryChangeEvidence, BinaryChangePredictor,
                            file_sha256, read_binary_pair)


class BinaryInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def image(self, name, width, height, rgb):
        path = self.root / name
        Image.new("RGB", (width, height), rgb).save(path)
        return path

    def test_request_order_and_native_normalization_are_preserved(self):
        first = self.image("first.png", 256, 256, (255, 0, 0))
        second = self.image("second.png", 256, 256, (0, 255, 0))
        a, b, hw = read_binary_pair(first, second)
        self.assertEqual(hw, (256, 256))
        torch.testing.assert_close(a[0, :, 0, 0], torch.tensor([1., -1., -1.]))
        torch.testing.assert_close(b[0, :, 0, 0], torch.tensor([-1., 1., -1.]))
        reversed_a, reversed_b, _ = read_binary_pair(second, first)
        self.assertTrue(torch.equal(reversed_a, b))
        self.assertTrue(torch.equal(reversed_b, a))

    def test_rectangular_pair_keeps_native_geometry_but_eval_tensor_is_256(self):
        a, b, hw = read_binary_pair(self.image("a.png", 512, 128, (127, 127, 127)),
                                    self.image("b.png", 512, 128, (128, 128, 128)))
        self.assertEqual(hw, (128, 512))
        self.assertEqual(a.shape, (1, 3, 256, 256))
        self.assertEqual(b.shape, a.shape)

    def test_mismatched_dimensions_keep_safe_specific_code(self):
        with self.assertRaisesRegex(ValueError, "^image_dimensions_mismatch$"):
            read_binary_pair(self.image("a.png", 256, 256, (0, 0, 0)),
                             self.image("b.png", 128, 256, (0, 0, 0)))

    def test_binary_evidence_cannot_be_mci_semantic_or_direction_evidence(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[140:150, 140:150] = 1
        result = BinaryChangeEvidence(mask, .1, (512, 512)).summary()
        self.assertEqual(result["mask_scope"], "generic_binary_change")
        self.assertEqual(result["mask_direction"], "not available")
        self.assertEqual(set(result["mask_grid"]), {"binary"})
        self.assertEqual(result["mask_grid"]["binary"]["changed_pixels"], 100)
        self.assertEqual(result["mask_grid"]["binary"]["region_pixels"]["lower_right"], 100)
        self.assertNotIn("caption", result)

    def test_malformed_mask_rejected_instead_of_silently_counting_other_classes(self):
        for mask in (np.full((4, 4), 2, np.uint8), np.zeros((4, 4), np.float32),
                     np.zeros((4, 4, 1), np.uint8)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises(ValueError):
                    BinaryChangeEvidence(mask, 0, (4, 4)).summary()


class StrictBinaryBundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = torch.nn.Linear(3, 2)

    def bundle(self, state):
        path = self.root / "weights.pt"
        torch.save(state, path)
        digest = file_sha256(path)
        manifest = {"architecture_id": ARCHITECTURE_ID, "source_revision": SOURCE_REVISION,
                    "vendored_network_sha256": NETWORK_SHA256,
                    "preprocessing": "author_eval_rgb_pil_bicubic_256_to_tensor_norm_0.5",
                    "model": {"input_nc": 3, "output_nc": 2, "embed_dim": 256, "decoder_softmax": False},
                    "checkpoint": {"path": "weights.pt", "size": path.stat().st_size, "sha256": digest}}
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return digest

    def construct(self, state):
        digest = self.bundle(state)
        with patch("tasks.change.binary_change.CHECKPOINT_SHA256", digest), \
                patch("tasks.change._binary.network.ChangeFormerV6", return_value=self.model):
            return BinaryChangePredictor(self.root)

    def test_all_parameters_loaded_strictly_frozen_without_unsafe_loader(self):
        state = {"weight": torch.full_like(self.model.weight, 7),
                 "bias": torch.full_like(self.model.bias, 3)}
        with patch("tasks.change.binary_change.torch.load", wraps=torch.load) as loader:
            predictor = self.construct(state)
        self.assertEqual(predictor.loading["matched_tensors"], 2)
        self.assertTrue(predictor.loading["strict_state"])
        self.assertTrue(torch.equal(self.model.weight, state["weight"]))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.parameters()))
        self.assertIs(loader.call_args.kwargs["weights_only"], True)

    def test_missing_and_extra_keys_cannot_be_compatibility_loaded(self):
        expected = self.model.state_dict()
        for state in ({"weight": expected["weight"]}, {**expected, "extra": torch.zeros(1)}):
            with self.subTest(keys=list(state)):
                with self.assertRaisesRegex(ValueError, "key mismatch"):
                    self.construct(state)

    def test_shape_dtype_and_nonfinite_tensors_rejected(self):
        expected = self.model.state_dict()
        for value in (torch.zeros(1), expected["weight"].double(),
                      torch.full_like(expected["weight"], float("nan"))):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    self.construct({**expected, "weight": value})


if __name__ == "__main__":
    unittest.main()
