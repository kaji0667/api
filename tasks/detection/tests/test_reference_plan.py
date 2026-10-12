"""Public synthetic grammar and relation-direction checks for the trial parser."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from tasks.detection.reference_plan import ReferencePlan, ReferenceSpec, parse_reference_plan


class ReferencePlanTests(unittest.TestCase):
    def require_plan(self, text: str) -> ReferencePlan:
        plan = parse_reference_plan(text)
        self.assertIsInstance(plan, ReferencePlan, text)
        assert plan is not None
        return plan

    def test_public_common_english_forms_preserve_reference_attributes(self):
        cases = (
            ("The baseball field is on the left of the orange and red ground track field in the middle",
             "The baseball field", "the orange and red ground track field in the middle", "left_of"),
            ("The baseball field is on the right of the tennis court on the left",
             "The baseball field", "the tennis court on the left", "right_of"),
            ("The vehicle is above the chimney at the bottom",
             "The vehicle", "the chimney at the bottom", "above"),
            ("The overpass is below the small vehicle",
             "The overpass", "the small vehicle", "below"),
        )
        for text, target, reference, relation in cases:
            with self.subTest(text=text):
                plan = self.require_plan(text)
                self.assertEqual(plan.target_description, target)
                self.assertEqual(plan.references, (ReferenceSpec(reference, relation),))
                self.assertIsNone(plan.absolute_position)

    def test_direct_english_relations(self):
        cases = (
            ("to the left of", "left_of"), ("left of", "left_of"),
            ("to the right of", "right_of"), ("right of", "right_of"),
            ("above", "above"), ("below", "below"),
            ("next to", "near"), ("near", "near"), ("beside", "near"),
        )
        for expression, relation in cases:
            with self.subTest(expression=expression):
                plan = self.require_plan(f"a striped courtyard {expression} a narrow road")
                self.assertEqual(plan.references, (ReferenceSpec("a narrow road", relation),))

    def test_english_and_comma_relations(self):
        plan = self.require_plan(
            "The lower-left lawn, to the left of a paved road and below a white building, next to a pond."
        )
        self.assertEqual(plan.target_description, "The lower-left lawn")
        self.assertEqual(plan.absolute_position, (-1, 1))
        self.assertEqual(plan.references, (
            ReferenceSpec("a paved road", "left_of"),
            ReferenceSpec("a white building", "below"),
            ReferenceSpec("a pond", "near"),
        ))

    def test_color_and_is_not_a_relation_separator(self):
        plan = self.require_plan(
            "the black and white building above the orange and red courtyard and next to a blue road"
        )
        self.assertEqual(plan.target_description, "the black and white building")
        self.assertEqual(plan.references, (
            ReferenceSpec("the orange and red courtyard", "above"),
            ReferenceSpec("a blue road", "near"),
        ))

    def test_relative_pronoun_on_target(self):
        plan = self.require_plan("a pale building that is to the left of a road, it is below a pond")
        self.assertEqual(plan.target_description, "a pale building")
        self.assertEqual(plan.references, (ReferenceSpec("a road", "left_of"), ReferenceSpec("a pond", "below")))

    def test_english_reverse_relation_directions(self):
        cases = (
            ("to the left of", "right_of"), ("to the right of", "left_of"),
            ("above", "below"), ("below", "above"), ("next to", "near"),
        )
        for expression, relation in cases:
            with self.subTest(expression=expression):
                plan = self.require_plan(f"a gray courtyard, a narrow road is {expression} it")
                self.assertEqual(plan.references, (ReferenceSpec("a narrow road", relation),))

    def test_english_reverse_relations_joined_by_and(self):
        plan = self.require_plan(
            "a gray courtyard, a road is to the right of it and an orange and red building is above the target"
        )
        self.assertEqual(plan.references, (
            ReferenceSpec("a road", "left_of"), ReferenceSpec("an orange and red building", "below"),
        ))

    def test_chinese_direct_relations(self):
        plan = self.require_plan("左下角的梯形草坪，在 道路 的左侧，位于白色建筑下方，靠近池塘")
        self.assertEqual(plan.target_description, "左下角的梯形草坪")
        self.assertEqual(plan.absolute_position, (-1, 1))
        self.assertEqual(plan.references, (
            ReferenceSpec("道路", "left_of"), ReferenceSpec("白色建筑", "below"), ReferenceSpec("池塘", "near"),
        ))

    def test_chinese_target_inline_relation(self):
        self.assertEqual(self.require_plan("草坪在道路右侧").references, (ReferenceSpec("道路", "right_of"),))
        self.assertEqual(self.require_plan("草坪靠近道路").references, (ReferenceSpec("道路", "near"),))

    def test_chinese_side_has_relations_are_inverted(self):
        plan = self.require_plan("右上角的庭院，左侧有道路，下方有建筑，旁边有池塘")
        self.assertEqual(plan.absolute_position, (1, -1))
        self.assertEqual(plan.references, (
            ReferenceSpec("道路", "right_of"), ReferenceSpec("建筑", "above"), ReferenceSpec("池塘", "near"),
        ))

    def test_chinese_reverse_relations(self):
        plan = self.require_plan("草坪，道路在它右侧，建筑位于目标上方")
        self.assertEqual(plan.references, (ReferenceSpec("道路", "left_of"), ReferenceSpec("建筑", "below")))

    def test_absolute_hint_comes_only_from_target(self):
        plan = self.require_plan("the lawn in the upper-left corner, below a building in the lower-right corner")
        self.assertEqual(plan.absolute_position, (-1, -1))
        self.assertEqual(plan.references[0].description, "a building in the lower-right corner")
        plan = self.require_plan("草坪，在左下角的建筑右侧")
        self.assertIsNone(plan.absolute_position)
        self.assertEqual(plan.references, (ReferenceSpec("左下角的建筑", "right_of"),))
        for text in ("a middle school above a road", "the center pivot irrigation field below a road"):
            with self.subTest(text=text):
                self.assertIsNone(self.require_plan(text).absolute_position)

    def test_nine_position_hints(self):
        cases = (
            ("a lawn in the top-left corner", (-1, -1)),
            ("a lawn in the top-right corner", (1, -1)),
            ("a lawn in the bottom-left corner", (-1, 1)),
            ("a lawn in the bottom-right corner", (1, 1)),
            ("a lawn at the top center of the image", (0, -1)),
            ("a lawn at the bottom center of the image", (0, 1)),
            ("a lawn in the middle-left region", (-1, 0)),
            ("a lawn in the middle-right region", (1, 0)),
            ("a lawn in the center of the image", (0, 0)),
        )
        for target, hint in cases:
            with self.subTest(target=target):
                plan = self.require_plan(target + ", next to a road")
                self.assertEqual(plan.target_description, target)
                self.assertEqual(plan.absolute_position, hint)

    def test_preserves_original_source_and_freezes_plan(self):
        source = "  An Unusual Facility, to the right of a Thin Road.  "
        plan = self.require_plan(source)
        self.assertEqual(plan.original_description, source)
        self.assertEqual(plan.target_description, "An Unusual Facility")
        self.assertEqual(plan.references[0].description, "a Thin Road")
        with self.assertRaises(FrozenInstanceError):
            plan.target_description = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            plan.references[0].relation = "below"  # type: ignore[misc]

    def test_declines_unsupported_or_ambiguous_grammar(self):
        cases = (
            "", "the bottom-left lawn", "草坪", "the lawn, a road",
            "the lawn is not above a road", "the lawn above no road",
            "草坪，不在道路左侧", "草坪，道路并非位于它右侧",
            "the lawn between a road and a building", "草坪，在道路和建筑之间",
            "the lawn behind a road", "草坪，在道路斜下方",
            "the lawn to the upper left of a road",
            "the lawn to the left of a building next to a road",
            "the lawn, above a road that is below a building",
            "a pale building that is to the left of a road, which is below a pond",
            "草坪，在道路左侧的建筑右侧",
            "the lawn and the courtyard above a road", "草坪和道路，在建筑左侧",
            "the lawn above a road and a building", "草坪，在道路和建筑左侧",
            "the lawn above it", "草坪，在它左侧",
            "the lawn left of itself", "the lawn left of target",
            "草坪，在自己左侧", "草坪，在目标左侧",
            "the lawn above a road, it is below a building with its courtyard",
            "the lawn, a road is to the right of a building",
            "the lower-left lawn in the upper-right corner, near a road",
            "the lawn above a road, below a road",
            "the lawn above the lawn",
            "the lawn above a road, below a building, near a pond, left of a path",
            "草坪，在道路左侧，位于建筑下方，靠近池塘，在小路上方",
            "the lawn, above the image", "草坪，在图像左侧",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(parse_reference_plan(text), text)


if __name__ == "__main__":
    unittest.main()
