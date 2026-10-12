"""Synthetic task text cases test preservation and explicit unsupported queries."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_task_text import TaskInputError, counting_text, detection_text, grounding_text, count_option_matches


class TaskTextTests(unittest.TestCase):
    def test_description_keeps_all_object_and_relationship_sentences(self):
        description = "The red warehouse at the lower left. A road is above it. Exclude the white building."
        question = "Return a pixel box only.\nDescription: " + description + "\nOnly output [x1,y1,x2,y2]."
        self.assertEqual(grounding_text(question), description)
        self.assertEqual(detection_text(question), description)

    def test_description_is_not_summarized_or_translated(self):
        description = "红色仓库位于左下方，右侧有道路。" + " Keep every qualifier." * 100
        self.assertEqual(grounding_text("Description: " + description), description)

    def test_multiple_descriptions_and_empty_payload_are_rejected(self):
        for value in ("Description: ", "Description: one\nDescription: two", " "):
            with self.subTest(value=value), self.assertRaises(TaskInputError):
                grounding_text(value)

    def test_presence_keeps_qualified_object_phrase(self):
        self.assertEqual(detection_text("Is there a red warehouse near the bridge in the image? Only answer Yes or No."),
                         "red warehouse near the bridge")
        self.assertEqual(detection_text("Are there any ships in this image?"), "ships")
        self.assertEqual(detection_text("Does this image contain a ship?"), "ship")

    def test_count_query_restores_original_class_name(self):
        for value in ("ship", "ships", "How many ships are in the image?",
                      "How many ships are there in the image? Only answer a number.",
                      "How many ships are there?", "How many ships do you see?",
                      "How many ships are visible?", "How many ships are present in this image?"):
            with self.subTest(value=value):
                self.assertEqual(counting_text(value), "ship")

    def test_qualified_count_extracts_target_category(self):
        cases = {
            "How many ships are in the lower left?": "ship",
            "How many ships are near the bridge?": "ship",
            "How many ships are in the lower left, except the red ships?": "ship",
            "How many ships are visible on the dock?": "ship",
            "How many ships on the dock?": "ship",
            "How many ships within the marked region?": "ship",
            "Within the upper-left region, how many ships are visible?": "ship",
            "How many small vehicles are in the lower-left corner?": "small vehicle",
            "How many storage tanks inside the marked region?": "storage tank",
            "How many container terminals are in the left half?": "container terminals",
            "How many red ships are in the image?": "ship",
            "How many ships\nare in the lower left?": "ship",
            "How many small\nvehicles are within the region?": "small vehicle",
            "Count the ships in the lower-left region.": "ship",
            "How many ships appear in the lower-left region?": "ship",
            "How many ships have been moored at the dock?": "ship",
            "How many ships (in the left region)?": "ship",
            "How many small red vehicles are in the image?": "small vehicle",
            "How many small airplanes are in the image?": "airplane",
            "How many large planes are in the image?": "airplane",
            "How many small aeroplanes are near the runway?": "airplane",
            "How many large vehicles are in the left half?": "large vehicle",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(counting_text(value), expected)

    def test_plain_query_extracts_target_category(self):
        for value in ("ship", "How many ships are in the image?", "How many ships are visible?",
                      "How many of the ships are in the image?", "What is the number of ships in the image?",
                      "Count the number of ships in the image.", "How many total ships are in the image?"):
            with self.subTest(value=value):
                self.assertEqual(counting_text(value), "ship")

    def test_equivalent_plane_words_recover_original_airplane_class(self):
        for value in ("plane", "planes", "airplane", "airplanes", "aeroplane", "aeroplanes",
                      "How many planes are in the upper-left corner?"):
            with self.subTest(value=value):
                self.assertEqual(counting_text(value), "airplane")

    def test_missing_and_multiple_count_targets_remain_explicit(self):
        for value in ("How many objects are there?", "How many are in the left region?",
                      "How many ships and vehicles are in the image?",
                      "How many ships are on the dock? How many trucks are in the image?"):
            with self.subTest(value=value), self.assertRaises(TaskInputError) as context:
                counting_text(value)
            self.assertEqual(context.exception.code, "unsupported_count_query")

    def test_count_class_labels_are_not_limited_to_a_static_target_list(self):
        for value in ("trainstation", "golf course", "truck", "container terminal"):
            with self.subTest(value=value):
                self.assertEqual(counting_text(value), value)
                self.assertEqual(counting_text("How many " + value + " are in the image?"), value)

    def test_numeric_range_options_keep_boundary_semantics(self):
        for value in ("3", "3 ships", "3艘", "1-3", "between 3 and 5", ">=3", "≤3",
                      "at least 3", "less than 4", "3+", "3 or fewer"):
            with self.subTest(value=value):
                self.assertTrue(count_option_matches(value, 3))
        for value in ("2", ">3", "<3", "4-9", "5-1", "approximately 3", "three ships"):
            with self.subTest(value=value):
                self.assertFalse(count_option_matches(value, 3))


if __name__ == "__main__":
    unittest.main()
