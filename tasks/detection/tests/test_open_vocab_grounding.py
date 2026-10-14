"""Contract checks for query roles and independent crop coordinate restoration."""
import unittest
import numpy as np
from tasks.detection.open_vocab_grounding import (
    box_iou, geometry_score, query_caption, restore_boxes, token_spans,
)


class OpenVocabularyContracts(unittest.TestCase):
    def test_references_never_enter_target_token_span(self):
        query = query_caption("the green lawn to the left of a road and below a building")
        self.assertEqual(query.parse_status,"target_reference_spans")
        self.assertEqual(query.caption,"the green lawn. a road. a building.")
        spans = token_spans([(0,0),(0,3),(4,9),(10,14),(14,15),(16,17),(18,22),(22,23),(24,25),(26,34),(34,35)],query.spans)
        self.assertEqual(spans,[[1,2,3],[5,6],[8,9]])

    def test_unsupported_description_preserves_all_conditions(self):
        text = "the building inside the park, with blue roofs and windows"
        query = query_caption(text)
        self.assertEqual(query.parse_status,"complete_description")
        self.assertEqual(query.plan.target_description,text)
        self.assertEqual(query.caption,text+".")

    def test_blank_and_over_budget_query_fail(self):
        for text in ("",None,"x"*10001):
            with self.assertRaises(ValueError):
                query_caption(text)

    def test_reference_corner_does_not_move_target(self):
        query = query_caption("the lawn near the building in the top left corner")
        self.assertIsNone(query.plan.absolute_position)
        self.assertIn("top left",query.plan.references[0].description)

    def test_unequal_axes_and_crop_origin_restored_once(self):
        boxes = restore_boxes(np.array([[.5,.5,.4,.2]]),(100,50,1100,450),(800,1600))
        np.testing.assert_allclose(boxes,[[400,210,800,290]])
        self.assertAlmostEqual(box_iou(boxes[0].tolist(),[400,210,800,290]),1.)

    def test_out_of_view_boxes_clip_to_original_not_square_input(self):
        boxes = restore_boxes(np.array([[.5,.5,2.,2.]]),(0,0,100,50),(50,100))
        np.testing.assert_array_equal(boxes,[[0,0,100,50]])

    def test_nonfinite_model_boxes_fail(self):
        with self.assertRaises(ValueError):
            restore_boxes(np.array([[.5,np.nan,.1,.1]]),(0,0,100,50),(50,100))

    def test_absolute_position_is_scored_in_original_image(self):
        query = query_caption("the lawn in the bottom left corner")
        self.assertGreater(geometry_score([0,80,20,100],query.plan,[],(100,100)),
                           geometry_score([80,0,100,20],query.plan,[],(100,100)))

    def test_empty_reference_is_neutral_and_no_candidate_is_invented(self):
        query = query_caption("the lawn near a road")
        self.assertEqual(geometry_score([0,0,10,10],query.plan,[{"candidates":[]}],(100,100)),.5)

    def test_missing_target_tokens_fail(self):
        with self.assertRaises(ValueError):
            token_spans([(0,0)],((0,4),))


if __name__ == "__main__":
    unittest.main()
