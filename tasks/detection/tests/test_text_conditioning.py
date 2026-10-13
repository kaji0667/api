"""Source attachment, full-text retention and normalized conditioning tests."""
import unittest
from unittest.mock import Mock

import torch

from tasks.detection.text_conditioning import parse_grounding_text, visual_description, RoleConditionedEncoder


class TextPlanTests(unittest.TestCase):
    def test_existing_roles_preserve_full_source_and_reference_attributes(self):
        source = "The small ship is below the large white ship on the right"
        plan = parse_grounding_text(source)
        self.assertEqual(plan.original_description, source)
        self.assertEqual(plan.target_description, "The small ship")
        self.assertEqual(plan.references[0].description, "the large white ship on the right")

    def test_multisentence_target_attributes_do_not_become_references(self):
        source = "The building. It has a gray roof. It is rectangular. A road is to the right of it."
        plan = parse_grounding_text(source)
        self.assertEqual(plan.target_description, "The building, a gray roof, rectangular")
        self.assertEqual(plan.references[0].description, "A road")
        self.assertEqual(plan.references[0].relation, "left_of")
        self.assertEqual(plan.original_description, source)

    def test_copular_attributes_are_target_properties(self):
        plan = parse_grounding_text("The vehicle is white and small.")
        self.assertEqual(plan.target_description, "The vehicle, white and small")
        self.assertFalse(plan.references)

    def test_unaccounted_negated_and_ambiguous_sentences_decline(self):
        for source in (
            "The building. It is gray. A road crosses a river.",
            "The building. It is not gray. A road is to the right of it.",
            "The building. Its neighbor is gray. A road is to the right of it.",
            "It is gray.", "The lawn above a road that is near a pond",
            "The building. It has a road. A road is to the right of it.",
        ):
            with self.subTest(source=source):
                self.assertIsNone(parse_grounding_text(source))

    def test_visual_crop_text_keeps_attributes_but_removes_frame_position(self):
        cases = (
            ("The small gray vehicle on the lower right", "The small gray vehicle"),
            ("a rectangular building in the top-left corner of the image", "a rectangular building"),
            ("the top-left red lawn", "the red lawn"),
            ("a bridge on the left", "a bridge"),
        )
        for source,expected in cases:
            self.assertEqual(visual_description(source),expected)
            self.assertEqual(parse_grounding_text(source).original_description,source)

    def test_visual_crop_text_preserves_ambiguous_attachment(self):
        for source in ("a left bank", "the middle school", "a building with a courtyard on the left",
                       "a bridge left of a road", "a building in the top-left corner on the bottom right"):
            self.assertEqual(visual_description(source),source)


class ConditionTests(unittest.TestCase):
    def setUp(self):
        self.full = torch.tensor([[1., 0., 0.]])
        self.cue = torch.tensor([[0., 1., 0.]])
        self.base = Mock()
        self.base.encode_descriptions.side_effect = [self.full, self.cue]

    def test_full_description_and_target_are_encoded_separately(self):
        source = "The ship below a road"
        result = RoleConditionedEncoder(self.base).encode_descriptions([source])
        self.assertEqual(self.base.encode_descriptions.call_args_list[0].args[0], [source])
        self.assertEqual(self.base.encode_descriptions.call_args_list[1].args[0], ["The ship"])
        expected = torch.tensor([[.75, .25, 0.]])
        expected /= expected.norm(dim=-1, keepdim=True)
        self.assertTrue(torch.equal(result, expected))
        self.assertTrue(torch.equal(self.full, torch.tensor([[1., 0., 0.]])))

    def test_uncertain_grammar_keeps_original_vector_bitwise(self):
        source = "The lawn between a road and a building"
        result = RoleConditionedEncoder(self.base).encode_descriptions([source])
        self.assertTrue(torch.equal(result, self.full))
        self.base.encode_descriptions.assert_called_once_with([source])

    def test_complete_text_limit_is_not_bypassed(self):
        self.base.encode_descriptions.side_effect = ValueError("text_context_exceeded")
        with self.assertRaisesRegex(ValueError, "text_context_exceeded"):
            RoleConditionedEncoder(self.base).encode_descriptions(["The ship below a road"])
        self.base.encode_descriptions.assert_called_once()

    def test_cancelling_or_nonfinite_vectors_cannot_reach_detector(self):
        for cue in (-self.full, torch.full_like(self.full, float("nan"))):
            self.base.encode_descriptions.side_effect = [self.full, cue]
            with self.assertRaises(ValueError):
                RoleConditionedEncoder(self.base, target_weight=.5).encode_descriptions(["The ship below a road"])


if __name__ == "__main__":
    unittest.main()
