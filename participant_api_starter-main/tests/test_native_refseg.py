"""CPU fixtures exercise native RefSeg contracts without loading actual weights."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api_task_text import TaskInputError
from dataset.rrsisd_refseg_dataset import RRSISDRefSegDataset
from native_refseg import NativeRefSegPredictor


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.load_state_dict = Mock()
        self.probabilities = torch.tensor([[[[0.2, 0.8], [0.2, 0.8]]]])
        self.forward_calls = []

    def forward(self, image, **kwargs):
        self.forward_calls.append((image.clone(), kwargs))
        return [torch.logit(self.probabilities)]


class NativeRefSegTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "model": {"synthetic": torch.tensor(1.0)},
            "args": {"imgsz": 512, "text_model_name": "ViT-L-14", "text_pretrained": "openai"},
            "data": {"nc": 20},
            "metrics": {"best_threshold": 0.55},
        }
        self.model = FakeModel()
        self.model_factory = Mock(return_value=self.model)
        self.encoder = Mock(embedding_dim=768)
        self.encoder.tokenizer = Mock(context_length=77)
        self.encoder.tokenizer.encode.return_value = [1] * 12
        self.encoder.encode.return_value = (torch.ones(1, 77, 768), torch.ones(1, 77, dtype=torch.bool))
        self.encoder_factory = Mock(return_value=self.encoder)

    def create(self, payload=None, **kwargs):
        with patch("native_refseg.resolve_checkpoint_path", return_value=Path("/synthetic/best_raw.pt")), patch(
            "native_refseg.torch.load", return_value=payload or self.payload
        ) as load:
            predictor = NativeRefSegPredictor(
                checkpoint="best_raw.pt", device="cpu", model_factory=self.model_factory,
                text_encoder_factory=self.encoder_factory, **kwargs
            )
            load.assert_called_once_with(Path("/synthetic/best_raw.pt"), map_location="cpu", weights_only=False)
        return predictor

    def test_original_model_state_eval_fp32_and_validation_threshold(self):
        predictor = self.create()
        self.model.load_state_dict.assert_called_once_with(self.payload["model"], strict=True)
        self.assertFalse(self.model.training)
        self.assertEqual(predictor.imgsz, 512)
        self.assertEqual(predictor.threshold, 0.55)
        self.assertEqual(self.model_factory.call_args.kwargs, {"ch": 3, "nc": 20, "verbose": False})
        self.encoder_factory.assert_called_once_with(
            model_name="ViT-L-14", pretrained="openai", device="cpu", precision="fp32", normalize=True
        )

    def test_explicit_size_override_saved_size_and_original_fallback(self):
        changed = copy.deepcopy(self.payload)
        changed["args"]["imgsz"] = 384
        self.assertEqual(self.create(changed).imgsz, 384)
        self.assertEqual(self.create(changed, imgsz=256).imgsz, 256)
        del changed["args"]["imgsz"]
        self.assertEqual(self.create(changed).imgsz, 512)

    def test_nonfinite_validation_threshold_is_rejected(self):
        changed = copy.deepcopy(self.payload)
        changed["metrics"]["best_threshold"] = float("nan")
        with self.assertRaises(TaskInputError) as error:
            self.create(changed)
        self.assertEqual(error.exception.code, "invalid_refseg_checkpoint")
        self.model_factory.assert_not_called()

    def test_noncanonical_checkpoint_is_rejected_before_loading(self):
        with patch("native_refseg.resolve_checkpoint_path", return_value=Path("/synthetic/best.pt")), patch(
            "native_refseg.torch.load"
        ) as load:
            with self.assertRaises(TaskInputError):
                NativeRefSegPredictor("best.pt", device="cpu")
            load.assert_not_called()

    def test_raw_question_is_parsed_only_inside_native_task(self):
        predictor = self.create()
        raw = " \tSynthetic wrapper.\nDescription: the red warehouse\n"
        with patch("native_refseg.grounding_text", return_value="the red warehouse") as parse:
            result = predictor.predict(Image.new("RGB", (10, 6), (255, 0, 0)), raw)
        parse.assert_called_once_with(raw)
        self.encoder.encode.assert_called_once_with(["the red warehouse"], batch_size=1, return_tokens=True)
        self.assertEqual(result.model_prompt, "the red warehouse")
        tensor, kwargs = self.model.forward_calls[0]
        self.assertEqual(tuple(tensor.shape), (1, 3, 512, 512))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.all(tensor[:, 0] == 1.0))
        self.assertTrue(torch.all(tensor[:, 1:] == 0.0))
        self.assertEqual(tuple(kwargs["text_embedding"].shape), (1, 77, 768))
        self.assertEqual(kwargs["text_token_mask"].dtype, torch.bool)
        self.assertEqual(result.original_size, (6, 10))
        self.assertEqual(result.mask.dtype, np.bool_)

    def test_over_budget_fails_before_encoding_or_model_forward(self):
        predictor = self.create()
        self.encoder.tokenizer.encode.return_value = [1] * 76
        with patch("native_refseg.grounding_text", return_value="synthetic long description"):
            with self.assertRaises(TaskInputError) as error:
                predictor.predict(Image.new("RGB", (10, 6)), "synthetic question")
        self.assertEqual(error.exception.code, "text_context_exceeded")
        self.encoder.encode.assert_not_called()
        self.assertEqual(self.model.forward_calls, [])

    def test_nonuniform_jpeg_decode_and_resize_match_original_dataset(self):
        predictor = self.create()
        y, x = np.indices((17, 29))
        pixels = np.stack(((x * 19 + y * 7) % 256, (y * 23 + x * 3) % 256, (x * y * 11) % 256), axis=-1)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "synthetic.jpg"
            Image.fromarray(pixels.astype(np.uint8)).save(image_path, quality=81, subsampling=2)
            # Only original image primitives are needed; no dataset/GT is loaded.
            with patch.object(RRSISDRefSegDataset, "__init__", side_effect=AssertionError("No dataset loading")):
                original = RRSISDRefSegDataset._read_image(str(image_path))
                resized, _ = RRSISDRefSegDataset._resize(
                    original, np.zeros(original.shape[:2], dtype=np.uint8), predictor.imgsz
                )
                np.testing.assert_array_equal(predictor._load_rgb(image_path), original)
                expected = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
                with patch("native_refseg.grounding_text", return_value="the warehouse"):
                    result = predictor.predict(image_path, "synthetic question")
        torch.testing.assert_close(self.model.forward_calls[0][0], expected, atol=0, rtol=0)
        self.assertEqual(result.original_size, (17, 29))

    def test_probability_restored_before_validation_threshold(self):
        changed = copy.deepcopy(self.payload)
        changed["metrics"]["best_threshold"] = 0.45
        predictor = self.create(changed)
        with patch("native_refseg.grounding_text", return_value="the warehouse"):
            result = predictor.predict(Image.new("RGB", (5, 2)), "synthetic question")
        expected = F.interpolate(
            self.model.probabilities, size=(2, 5), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        np.testing.assert_allclose(result.probability, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(result.mask, expected > 0.45)
        threshold_first = F.interpolate(
            (self.model.probabilities > 0.45).float(), size=(2, 5), mode="nearest"
        )[0, 0].numpy().astype(bool)
        self.assertFalse(np.array_equal(result.mask, threshold_first))

    def test_invalid_token_mask_and_nonfinite_logits_fail_explicitly(self):
        predictor = self.create()
        self.encoder.encode.return_value = (torch.ones(1, 77, 768), torch.zeros(1, 77, dtype=torch.bool))
        with patch("native_refseg.grounding_text", return_value="the warehouse"):
            with self.assertRaises(TaskInputError) as error:
                predictor.predict(Image.new("RGB", (5, 2)), "synthetic question")
        self.assertEqual(error.exception.code, "invalid_refseg_text_features")
        self.assertEqual(self.model.forward_calls, [])
        self.encoder.encode.return_value = (torch.ones(1, 77, 768), torch.ones(1, 77, dtype=torch.bool))
        self.model.probabilities = torch.full((1, 1, 2, 2), float("nan"))
        with patch("native_refseg.grounding_text", return_value="the warehouse"):
            with self.assertRaises(TaskInputError) as error:
                predictor.predict(Image.new("RGB", (5, 2)), "synthetic question")
        self.assertEqual(error.exception.code, "invalid_refseg_result")


if __name__ == "__main__":
    unittest.main()
