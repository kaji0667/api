"""Qwen adapter contract tests; these do not load weights or require a GPU."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_adapter import parse_generated_answer, prompt_for_request  # noqa: E402


def request(kind: str) -> dict:
    value = {
        "question": "Question?",
        "images": [{"asset_id": "a" * 32}],
        "response_constraint": {"type": kind},
        "answer_type": "short_text",
    }
    if kind == "bbox":
        value.update({"answer_type": "bbox", "image_width": 200, "image_height": 100})
    if kind == "enum":
        value["response_constraint"].update({"values": ["Yes", "No"], "question_form": "yes_no"})
    if kind == "single_choice":
        value["response_constraint"]["values"] = ["A", "B"]
        value["choices"] = {"A": "Road", "B": "River"}
        value["answer_type"] = "single_choice"
    return value


class QwenAdapterTests(unittest.TestCase):
    def test_bbox_prompt_and_conversion(self) -> None:
        item = request("bbox")
        prompt, token_class = prompt_for_request(item, "normalized_1000")
        self.assertIn("normalized 0-1000 grid", prompt)
        self.assertEqual(token_class, "bbox")
        self.assertEqual(
            parse_generated_answer("[100, 200, 900, 800]", item, "normalized_1000"),
            [20, 20, 180, 80],
        )
        self.assertEqual(
            parse_generated_answer("[100,\n200,\n900,\n800]", item, "normalized_1000"),
            [20, 20, 180, 80],
        )
        self.assertEqual(
            parse_generated_answer("[20, 20, 180, 80]", item, "pixel_xyxy"),
            [20, 20, 180, 80],
        )

    def test_rejects_bad_bbox(self) -> None:
        item = request("bbox")
        for raw in ("[900, 0, 100, 500]", "[0, 0, 1001, 500]", "not JSON"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_generated_answer(raw, item, "normalized_1000")

    def test_enum_choice_integer_and_short_text(self) -> None:
        yes_no = request("enum")
        self.assertEqual(prompt_for_request(yes_no, "normalized_1000")[1], "yes_no")
        self.assertEqual(parse_generated_answer("yes", yes_no, "normalized_1000"), "Yes")
        self.assertEqual(parse_generated_answer("B", request("single_choice"), "normalized_1000"), "B")
        self.assertEqual(parse_generated_answer("0", request("integer"), "normalized_1000"), "0")
        self.assertEqual(parse_generated_answer("bridge", request("short_text"), "normalized_1000"), "bridge")
        with self.assertRaises(ValueError):
            parse_generated_answer("<think>unfinished", request("short_text"), "normalized_1000")
        with self.assertRaises(ValueError):
            parse_generated_answer("The answer is B", request("single_choice"), "normalized_1000")

    def test_temporal_order_and_thinking_suffix(self) -> None:
        item = request("enum")
        item["images"].append({"asset_id": "b" * 32})
        prompt, _ = prompt_for_request(item, "normalized_1000")
        self.assertIn("first image is before/Image A", prompt)
        self.assertEqual(
            parse_generated_answer("<think>hidden</think>Yes", item, "normalized_1000"),
            "Yes",
        )


if __name__ == "__main__":
    unittest.main()
