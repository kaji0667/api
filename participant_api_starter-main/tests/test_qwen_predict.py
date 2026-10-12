"""Exercise the Qwen adapter's orchestration without torch, weights, or a GPU."""

import contextlib
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_adapter  # noqa: E402
import qwen_adapter  # noqa: E402


class FakeImage:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    def copy(self):
        return self

    def close(self):
        self.closed = True


class FakeOpened:
    def __init__(self, name: str, collected: list[FakeImage]):
        self.name = name
        self.collected = collected

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def convert(self, mode: str):
        assert mode == "RGB"
        image = FakeImage(self.name)
        self.collected.append(image)
        return image


class FakeTensor:
    shape = (1, 3)

    def __init__(self, floating: bool):
        self.floating = floating
        self.move_kwargs = None

    def is_floating_point(self):
        return self.floating

    def to(self, **kwargs):
        self.move_kwargs = kwargs
        return self


class FakeProcessor:
    def __init__(self, output: str):
        self.output = output
        self.messages = None
        self.inputs = {"input_ids": FakeTensor(False), "pixel_values": FakeTensor(True)}

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs["enable_thinking"] is False
        assert kwargs["return_tensors"] == "pt"
        return self.inputs

    def decode(self, new_ids, **kwargs):
        assert new_ids == ["generated"]
        assert kwargs["skip_special_tokens"] is True
        return self.output


class FakeOutput:
    def __getitem__(self, key):
        assert key == (0, slice(3, None))
        return ["generated"]


class FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return FakeOutput()


class QwenPredictTests(unittest.TestCase):
    def make_adapter(self, output: str):
        adapter = object.__new__(qwen_adapter.QwenModelAdapter)
        images = []
        adapter.bbox_mode = "normalized_1000"
        adapter.Image = SimpleNamespace(open=lambda path: FakeOpened(path.name, images))
        adapter.processor = FakeProcessor(output)
        adapter.model = FakeModel()
        adapter.torch = SimpleNamespace(bfloat16="bf16", inference_mode=contextlib.nullcontext)
        adapter.lock = threading.Lock()
        return adapter, images

    def test_two_images_keep_order_and_return_enum(self):
        adapter, images = self.make_adapter("Yes")
        request = {
            "question": "Did the scene change?",
            "images": [{"asset_id": "a"}, {"asset_id": "b"}],
            "response_constraint": {"type": "enum", "values": ["Yes", "No"], "question_form": "yes_no"},
            "answer_type": "short_text",
        }
        answer = adapter.predict(request, [Path("before.png"), Path("after.png")])
        self.assertEqual(answer, "Yes")
        content = adapter.processor.messages[1]["content"]
        self.assertEqual([part["image"].name for part in content[:-1]], ["before.png", "after.png"])
        self.assertIn("first image is before/Image A", content[-1]["text"])
        self.assertEqual(adapter.model.generate_kwargs["max_new_tokens"], 8)
        self.assertEqual(adapter.processor.inputs["pixel_values"].move_kwargs["dtype"], "bf16")
        self.assertNotIn("dtype", adapter.processor.inputs["input_ids"].move_kwargs)
        self.assertTrue(all(image.closed for image in images))

    def test_bbox_becomes_original_pixel_coordinates(self):
        adapter, images = self.make_adapter("[100,200,900,800]")
        request = {
            "question": "Where is the object?",
            "images": [{"asset_id": "a"}],
            "response_constraint": {"type": "bbox"},
            "answer_type": "bbox", "image_width": 200, "image_height": 100,
        }
        answer = adapter.predict(request, [Path("image.png")])
        self.assertEqual(answer, [20, 20, 180, 80])
        self.assertTrue(images[0].closed)

    def test_build_model_selects_optional_qwen_adapter(self):
        sentinel = object()
        with patch.dict("os.environ", {"MODEL_MODE": "qwen"}), patch.object(
            qwen_adapter, "QwenModelAdapter", return_value=sentinel
        ):
            self.assertIs(model_adapter.build_model(), sentinel)


if __name__ == "__main__":
    unittest.main()
