"""Grounding protocol formatting must not become object/relationship text."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from api_task_text import grounding_text,detection_text,TaskInputError
from tasks.detection.reference_plan import parse_reference_plan


DIRECTIVES=(
    "Return a bbox.",
    "Return the bounding box as [x1,y1,x2,y2].",
    "Please provide the bounding box in pixel coordinates.",
    "Report box coordinates only.",
    "Your response must be a bbox.",
    "请只返回坐标，不要包含解释。",
)


class BboxWrappingTests(unittest.TestCase):
    def test_box_directives_leave_every_target_and_reference_clause_intact(self):
        description="The lower-left lawn\nA road is to the right of it"
        for directive in DIRECTIVES:
            for marker in ("Description:","Description：","目标描述：","描述:"):
                question=marker+"\n"+description+"\n"+directive
                with self.subTest(directive=directive,marker=marker):
                    self.assertEqual(grounding_text(question),description)
                    self.assertEqual(detection_text(question),description)
                    self.assertEqual(parse_reference_plan(detection_text(question)),parse_reference_plan(description))

    def test_same_line_instruction_ends_at_sentence_boundary(self):
        source="The gray warehouse. A road is above it. Exclude the white building."
        for directive in DIRECTIVES:
            self.assertEqual(grounding_text("Description: "+source+" "+directive),source)

    def test_visual_attributes_and_negative_constraints_are_not_formatting(self):
        descriptions=("The box-shaped gray building with a roof on the left.",
                      "The ship does not return to the harbor. A road is above it.",
                      "The lawn without the white building.",
                      "坐标轴图案附近的灰色建筑，排除白色屋顶的建筑。")
        for description in descriptions:
            self.assertEqual(grounding_text("Description: "+description),description)

    def test_unknown_instruction_is_not_silently_deleted(self):
        source="The building. Choose the nicest-looking object."
        self.assertEqual(grounding_text("Description: "+source),source)

    def test_ambiguous_multiple_payloads_are_rejected_across_languages(self):
        for source in ("Description: one\n目标描述：two","描述：one\nDescription: two"):
            with self.assertRaises(TaskInputError):
                grounding_text(source)


if __name__=="__main__":
    unittest.main()
