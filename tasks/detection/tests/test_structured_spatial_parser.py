"""Synthetic source, scope and local HTTP contract checks; no evaluation text."""
import json
import unittest
from unittest.mock import Mock, patch
import urllib.error

from tasks.detection.structured_spatial_parser import (
    QwenSpatialPlanParser, SpatialParseError, source_sentences, validate_spatial_plan,
    _complete_literal_relations,
)


def position(name="none", source=""):
    return dict(position=name, source=source)


def record(target, **changes):
    return dict(target_parts=[target], target_source_id=0, global_position=position(),
                references=[], containers=[], frame_edges=[], unresolved=[]) | changes


def reference(description, source_id, direction, location=None):
    return dict(description=description, source_id=source_id, direction=direction,
                global_position=location or position())


class SourcePlanTests(unittest.TestCase):
    text = ("The area to be located is the blue rectangular shed at the bottom left of the image. "
            "Above it is a narrow paved lane, to the left is a gravel yard, "
            "below it is a green lawn, and to the right are two white tanks.")

    def plan(self):
        return record("the blue rectangular shed",
                      global_position=position("bottom_left", "bottom left of the image"),
                      references=[reference(description, 1, direction) for description, direction in (
                          ("a narrow paved lane", "above"), ("a gravel yard", "left"),
                          ("a green lawn", "below"), ("two white tanks", "right"),
                      )])

    def rejected(self, value, text, code):
        with self.assertRaises(SpatialParseError) as raised:
            validate_spatial_plan(value, text)
        self.assertEqual(raised.exception.code, code)

    def test_four_neighbors_keep_comma_and_invert_relation(self):
        self.assertEqual(len(source_sentences(self.text)), 2)
        plan = validate_spatial_plan(self.plan(), self.text)
        self.assertEqual(plan.absolute_position, (-1, 1))
        self.assertEqual(plan.target_query, "the blue rectangular shed")
        self.assertEqual([r.relation for r in plan.references], ["below", "right_of", "above", "left_of"])
        self.assertEqual(plan.references[-1].description, "two white tanks")

    def test_articles_are_restored_from_source(self):
        value = self.plan()
        value["target_parts"] = ["a blue rectangular shed."]
        self.assertEqual(validate_spatial_plan(value, self.text).target_query, "the blue rectangular shed")

    def test_shared_direction_has_separate_references(self):
        text = "The target is a red hut. Above it are a lane and a white tower."
        value = record("a red hut", references=[reference("a lane", 1, "above"),
                                               reference("a white tower", 1, "above")])
        self.assertEqual(len(validate_spatial_plan(value, text).references), 2)

    def test_diagonal_reference_becomes_two_constraints(self):
        text = "The target is a red hut. To its upper right is a white tower."
        value = record("a red hut", references=[reference("a white tower", 1, "upper_right")])
        self.assertEqual(validate_spatial_plan(value, text).references[0].relations, ("left_of", "below"))

    def test_literal_direction_and_inversion_override_wrong_model_role_label(self):
        for text, direction, relation in (("The target is a red hut. Above it is a white tower.", "below", "below"),
                                          ("The target is a red hut. It is left of a white tower.", "left", "left_of")):
            value = record("a red hut", references=[reference("a white tower", 1, direction)])
            self.assertEqual(validate_spatial_plan(value, text).references[0].relation, relation)

    def test_conjoined_target_relations_each_use_their_own_direction(self):
        text = "The target is a red hut. It is left of a white tower and below a blue shed."
        value = record("a red hut", references=[reference("a white tower", 1, "left"),
                                               reference("a blue shed", 1, "below")])
        self.assertEqual([r.relation for r in validate_spatial_plan(value, text).references], ["left_of", "below"])

    def test_container_position_is_local_and_later_attributes_are_preserved(self):
        text = ("The target is the only yellow van in the upper-left part of a parking lot. "
                "The parking lot is on the right side of the image. The van has a black roof.")
        value = record("the only yellow van", target_parts=["the only yellow van", "a black roof"],
                       containers=[dict(description="a parking lot", source_id=0,
                                        global_position=position("right", "right side of the image"),
                                        target_position=position("top_left", "upper-left part of a parking lot"))])
        plan = validate_spatial_plan(value, text)
        self.assertIsNone(plan.absolute_position)
        self.assertEqual(plan.containers[0].absolute_position, (1, None))
        self.assertEqual(plan.containers[0].target_position, (-1, -1))
        self.assertIn("only", plan.target_query)
        self.assertIn("black roof", plan.target_query)

    def test_frame_edge_is_a_constraint_without_detectable_reference(self):
        text = "The target is a red hut. It is adjacent to the left edge of the image."
        value = record("a red hut", frame_edges=[dict(side="left", source_id=1)])
        plan = validate_spatial_plan(value, text)
        self.assertEqual(plan.frame_edges, ("left",))
        self.assertEqual(plan.absolute_position, (-1, None))
        self.assertEqual(plan.references, ())

    def test_simple_query_has_no_fake_spatial_constraint(self):
        plan = validate_spatial_plan(record("sheds"), "sheds")
        self.assertIsNone(plan.absolute_position)
        self.assertFalse(plan.references or plan.containers or plan.frame_edges)

    def test_existing_explicit_chinese_prefix_is_preserved_and_removed_from_identity(self):
        text = "左下角的梯形草坪。\n上方有红色道路。"
        value = record("左下角的梯形草坪", references=[reference("红色道路", 1, "above")])
        plan = validate_spatial_plan(value, text)
        self.assertEqual(plan.absolute_position, (-1, 1))
        self.assertEqual(plan.target_query, "梯形草坪")

    def test_fabricated_object_is_rejected(self):
        value = self.plan()
        value["references"][0]["description"] = "a black runway"
        self.rejected(value, self.text, "grounding_parser_source_mismatch")

    def test_missing_attribute_quantity_or_whole_clause_is_rejected(self):
        for change in (lambda v: v["target_parts"].__setitem__(0, "shed"),
                       lambda v: v["references"][-1].__setitem__("description", "white tanks"),
                       lambda v: v["references"].pop()):
            value = self.plan()
            change(value)
            self.rejected(value, self.text, "grounding_parser_unaccounted_text")

    def test_missing_later_attribute_is_rejected(self):
        self.rejected(record("a red hut"), "The target is a red hut. It has a striped roof.",
                      "grounding_parser_unaccounted_text")

    def test_bad_source_id_and_extra_fields_are_rejected(self):
        for index in (-1, 7, True, "1"):
            value = self.plan()
            value["references"][0]["source_id"] = index
            self.rejected(value, self.text, "grounding_parser_source_mismatch")
        value = self.plan() | {"bbox": [0, 0, 1, 1]}
        self.rejected(value, self.text, "grounding_parser_invalid_output")

    def test_position_must_have_direction_evidence(self):
        value = self.plan()
        value["global_position"]["position"] = "top_right"
        self.rejected(value, self.text, "grounding_parser_source_mismatch")

    def test_reference_cannot_copy_target_position(self):
        value = self.plan()
        value["references"][0]["global_position"] = value["global_position"]
        self.rejected(value, self.text, "grounding_parser_source_mismatch")

    def test_image_is_never_a_container(self):
        value = self.plan()
        value["containers"] = [dict(description="the image", source_id=0,
                                    global_position=position(), target_position=position())]
        self.rejected(value, self.text, "grounding_parser_invalid_output")

    def test_unknown_or_negative_relation_is_explicitly_rejected(self):
        text = "The target is a red hut. It is not above a white tower."
        value = record("a red hut", references=[reference("a white tower", 1, "below")])
        self.rejected(value, text, "grounding_parser_unresolved")
        self.rejected(record("a red hut", unresolved=[1]), text, "grounding_parser_unresolved")

    def test_reference_edge_cannot_be_applied_to_target(self):
        text = "The target is a red hut. Above it is a tower close to the right edge of the image."
        value = record("a red hut", references=[reference("a tower close to the right edge of the image", 1, "above")],
                       frame_edges=[dict(side="right", source_id=1)])
        self.rejected(value, text, "grounding_parser_unresolved")

    def test_target_is_not_its_own_container(self):
        text = "The target is a red hut in the top-left corner of the image."
        value = record("a red hut", global_position=position("top_left", "top-left corner of the image"),
                       containers=[dict(description="a red hut", source_id=0,
                                        global_position=position(), target_position=position())])
        self.rejected(value, text, "grounding_parser_invalid_output")

    def test_image_position_cannot_be_container_local_position(self):
        text = "The target is a car in a parking lot on the right side of the image."
        value = record("a car", containers=[dict(description="a parking lot", source_id=0,
                                               global_position=position(),
                                               target_position=position("right", "right side of the image"))])
        self.rejected(value, text, "grounding_parser_conflicting_position")


class SourceClauseBoundaryTests(unittest.TestCase):
    """Synthetic clause ownership without private requests or model calls."""

    def assert_rejected(self, value, text):
        with self.assertRaises(SpatialParseError):
            validate_spatial_plan(value, text)

    def test_omitted_same_name_neighbor_in_another_direction_is_rejected(self):
        text = ("The target is a coral hexagonal kiosk. "
                "Above it is a silver bench, and below it is a silver bench.")
        value = record("a coral hexagonal kiosk",
                       references=[reference("a silver bench", 1, "above")])
        # A word-set coverage check alone cannot see the omitted second bench.
        self.assert_rejected(value, text)

    def test_same_name_neighbors_keep_their_distinct_source_directions(self):
        text = ("The target is a coral hexagonal kiosk. "
                "Above it is a silver bench, and below it is a silver bench.")
        value = record("a coral hexagonal kiosk", references=[
            reference("a silver bench", 1, "above"),
            reference("a silver bench", 1, "below"),
        ])
        plan = validate_spatial_plan(value, text)
        self.assertEqual([ref.relation for ref in plan.references], ["below", "above"])
        self.assertEqual([ref.description for ref in plan.references],
                         ["a silver bench", "a silver bench"])

    def test_shared_direction_second_neighbor_cannot_keep_a_wrong_label(self):
        text = ("The target is a coral hexagonal kiosk. "
                "Above it are a narrow footpath and a bronze statue.")
        value = record("a coral hexagonal kiosk", references=[
            reference("a narrow footpath", 1, "above"),
            reference("a bronze statue", 1, "below"),
        ])
        try:
            plan = validate_spatial_plan(value, text)
        except SpatialParseError:
            # Conservative rejection preserves the caller's original fallback.
            return
        self.assertEqual([ref.relation for ref in plan.references], ["below", "below"])

    def test_complete_attached_noun_phrases_are_preserved(self):
        descriptions = (
            "a cobbled footpath lined with tall reeds",
            "a slate tower with a striped dome",
            "a silver pavilion that has a glass roof",
        )
        for description in descriptions:
            with self.subTest(description=description):
                text = "The target is a coral hexagonal kiosk. Above it is " + description + "."
                value = record("a coral hexagonal kiosk",
                               references=[reference(description, 1, "above")])
                plan = validate_spatial_plan(value, text)
                self.assertEqual(len(plan.references), 1)
                self.assertEqual(plan.references[0].description, description)
                self.assertEqual(plan.references[0].relation, "below")

    def test_attached_modifiers_cannot_be_split_into_independent_neighbors(self):
        cases = (
            ("a cobbled footpath lined with tall reeds", "a cobbled footpath", "tall reeds"),
            ("a slate tower with a striped dome", "a slate tower", "a striped dome"),
            ("a silver pavilion that has a glass roof", "a silver pavilion", "a glass roof"),
        )
        for description, head, detail in cases:
            with self.subTest(description=description):
                text = "The target is a coral hexagonal kiosk. Above it is " + description + "."
                value = record("a coral hexagonal kiosk", references=[
                    reference(head, 1, "above"), reference(detail, 1, "above"),
                ])
                self.assert_rejected(value, text)

    def test_top_level_coordination_keeps_modifiers_on_their_noun(self):
        text = ("The target is a coral hexagonal kiosk. "
                "Above it are a bronze statue and a cobbled footpath lined with tall reeds.")
        value = record("a coral hexagonal kiosk", references=[
            reference("a bronze statue", 1, "above"),
            reference("a cobbled footpath lined with tall reeds", 1, "above"),
        ])
        plan = validate_spatial_plan(value, text)
        self.assertEqual([ref.description for ref in plan.references],
                         ["a bronze statue", "a cobbled footpath lined with tall reeds"])
        self.assertEqual([ref.relation for ref in plan.references], ["below", "below"])

    def test_while_clause_target_frame_edge_shares_the_original_sentence_id(self):
        text = ("The target is a coral hexagonal kiosk. Above it is a silver bench, "
                "while the target touches the right edge of the image.")
        self.assertEqual(len(source_sentences(text)), 2)
        value = record("a coral hexagonal kiosk",
                       references=[reference("a silver bench", 1, "above")],
                       frame_edges=[dict(side="right", source_id=1)])
        plan = validate_spatial_plan(value, text)
        self.assertEqual(plan.frame_edges, ("right",))
        self.assertEqual(plan.absolute_position, (1, None))
        self.assertEqual(plan.references[0].relation, "below")

    def test_reference_own_image_edge_cannot_be_applied_to_target(self):
        text = ("The target is a coral hexagonal kiosk. Above it is a bronze statue "
                "near the left edge of the image.")
        value = record("a coral hexagonal kiosk", references=[
            reference("a bronze statue near the left edge of the image", 1, "above"),
        ], frame_edges=[dict(side="left", source_id=1)])
        self.assert_rejected(value, text)

    def test_an_objects_edge_is_not_an_image_edge(self):
        text = ("The target is a coral hexagonal kiosk. "
                "Its flat roof touches the left edge of the kiosk.")
        value = record("a coral hexagonal kiosk",
                       target_parts=["a coral hexagonal kiosk", "flat roof"],
                       frame_edges=[dict(side="left", source_id=1)])
        self.assert_rejected(value, text)

    def test_frame_edge_side_requires_literal_evidence(self):
        text = ("The target is a coral hexagonal kiosk. "
                "It touches the left edge of the image.")
        value = record("a coral hexagonal kiosk",
                       frame_edges=[dict(side="right", source_id=1)])
        self.assert_rejected(value, text)

    def test_literal_target_image_edge_remains_supported(self):
        text = ("The target is a coral hexagonal kiosk. "
                "It touches the left edge of the image.")
        value = record("a coral hexagonal kiosk",
                       frame_edges=[dict(side="left", source_id=1)])
        plan = validate_spatial_plan(value, text)
        self.assertEqual(plan.frame_edges, ("left",))
        self.assertEqual(plan.absolute_position, (-1, None))
        self.assertEqual(plan.references, ())


class ParserHTTPTests(unittest.TestCase):
    def call(self, outer, text="sheds"):
        response = Mock()
        response.read.return_value = json.dumps(outer).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch("urllib.request.build_opener", return_value=opener):
            plan = QwenSpatialPlanParser()(text)
        return plan, opener.open.call_args

    def test_only_local_schema_constrained_text_is_sent(self):
        plan, args = self.call(dict(done=True, message=dict(content=json.dumps(record("sheds")))))
        request = args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(plan.target_query, "sheds")
        self.assertFalse(payload["think"] or payload["stream"])
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertIn("sentences", payload["messages"][-1]["content"])
        self.assertNotIn("images", payload)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")

    def test_incomplete_model_neighbors_are_completed_from_each_source_clause(self):
        text = ("The target is a purple circular gazebo. To its right, there are a gravel path and a blue pond. "
                "Below it is a wooden fence, while the top is bordered by the edge of the image. "
                "To its left are a blue pond and a gravel path.")
        raw = record("a purple circular gazebo", references=[
            reference("a gravel path", 1, "right"), reference("a wooden fence", 2, "below"),
            reference("a blue pond", 3, "left"),
        ])
        plan, _ = self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)
        self.assertEqual([(r.description, r.relation) for r in plan.references], [
            ("a gravel path", "left_of"), ("a blue pond", "left_of"),
            ("a wooden fence", "above"), ("a blue pond", "right_of"), ("a gravel path", "right_of"),
        ])
        self.assertEqual(plan.frame_edges, ("top",))
        self.assertEqual(plan.absolute_position, (None, -1))

    def test_clause_completion_keeps_color_coordination_and_attached_entities(self):
        for description in ("a red and white canopy", "a garden with trees and shrubs",
                            "a lane lined with two trees and three shrubs"):
            with self.subTest(description=description):
                text = "The target is a purple gazebo. Above it is " + description + "."
                raw = record("a purple gazebo")
                plan, _ = self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)
                self.assertEqual([r.description for r in plan.references], [description])

    def test_comma_inside_an_attribute_does_not_split_a_neighbor(self):
        description = "a small, green pond"
        text = "The target is a purple gazebo. Above it is " + description + "."
        raw = record("a purple gazebo")
        plan, _ = self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)
        self.assertEqual([r.description for r in plan.references], [description])

    def test_completion_cannot_erase_invalid_source_positions(self):
        text = "The target is a purple gazebo. Above it is a tower with two domes."
        for location in (position("invented", ""), position("right", "right side of the image"),
                         position("top", "Above it")):
            with self.subTest(location=location), self.assertRaises(SpatialParseError):
                raw = record("a purple gazebo", references=[reference("a tower", 1, "above", location)])
                self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)

    def test_completion_preserves_source_bound_explicit_reference_position(self):
        text = ("The target is a purple gazebo. Above it is a tower with two domes "
                "on the right side of the image.")
        raw = record("a purple gazebo", references=[reference("a tower", 1, "above",
                         position("right", "right side of the image"))])
        plan, _ = self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)
        self.assertEqual(plan.references[0].absolute_position, (1, None))
        self.assertIn("two domes", plan.references[0].description)

    def test_one_coordinated_neighbor_cannot_borrow_the_others_image_position(self):
        text = ("The target is a purple gazebo. Above it are a pond and a tower "
                "on the right side of the image.")
        raw = record("a purple gazebo", references=[reference("a pond", 1, "above",
                         position("right", "right side of the image"))])
        with self.assertRaises(SpatialParseError):
            self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)
        direct = record("a purple gazebo", references=[
            reference("a pond", 1, "above", position("right", "right side of the image")),
            reference("a tower on the right side of the image", 1, "above"),
        ])
        with self.assertRaises(SpatialParseError):
            validate_spatial_plan(direct, text)

    def test_coordinated_neighbors_keep_their_own_explicit_position(self):
        text = ("The target is a purple gazebo. Above it are a pond and a tower "
                "on the right side of the image.")
        raw = record("a purple gazebo", references=[reference("a tower", 1, "above",
                         position("right", "right side of the image"))])
        plan, _ = self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)
        self.assertIsNone(plan.references[0].absolute_position)
        self.assertEqual(plan.references[1].absolute_position, (1, None))

    def test_ambiguous_edge_pronoun_is_not_completed_as_target_edge(self):
        text = ("The target is a purple gazebo. Above it is a tower, "
                "while it touches the right edge of the image.")
        for edges in ([], [dict(side="right", source_id=1)]):
            with self.subTest(edges=edges), self.assertRaises(SpatialParseError):
                raw = record("a purple gazebo", references=[reference("a tower", 1, "above")],
                             frame_edges=edges)
                self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)

    def test_partial_frame_clause_cannot_discard_a_second_edge_or_constraint(self):
        for suffix in ("and the right edge of the image", "with a blue pond nearby"):
            text = "The target is a purple gazebo. It touches the left edge of the image " + suffix + "."
            with self.subTest(suffix=suffix), self.assertRaises(SpatialParseError):
                self.call(dict(done=True, message=dict(content=json.dumps(record("a purple gazebo")))), text)

    def test_completion_does_not_remove_unresolved_or_negative_constraints(self):
        for text, raw in (
            ("The target is a purple gazebo. Above it is a tower.", record("a purple gazebo", unresolved=[1])),
            ("The target is a purple gazebo. Above it is a tower without a dome.", record("a purple gazebo")),
        ):
            with self.subTest(text=text), self.assertRaises(SpatialParseError):
                self.call(dict(done=True, message=dict(content=json.dumps(raw))), text)

    def test_dependency_parser_unavailable_has_no_silent_coordination_guess(self):
        text = "The target is a purple gazebo. Above it are a pond and a tower."
        with patch("tasks.detection.structured_spatial_parser._english_dependencies",
                   side_effect=SpatialParseError("grounding_parser_unavailable")), self.assertRaises(SpatialParseError) as raised:
            _complete_literal_relations(record("a purple gazebo"), text)
        self.assertEqual(raised.exception.code, "grounding_parser_unavailable")

    def test_invalid_or_incomplete_response_has_fixed_error(self):
        for outer in (dict(done=False), [], dict(done=True, message=dict(content="{"))):
            with self.assertRaises(SpatialParseError) as raised:
                self.call(outer)
            self.assertEqual(raised.exception.code, "grounding_parser_invalid_output")

    def test_unavailable_parser_does_not_expose_error_text(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.URLError("private-input")
        with patch("urllib.request.build_opener", return_value=opener), self.assertRaises(SpatialParseError) as raised:
            QwenSpatialPlanParser()("sheds")
        self.assertEqual(str(raised.exception), "grounding_parser_unavailable")

    def test_remote_endpoint_and_invalid_timeout_are_rejected(self):
        for url in ("https://127.0.0.1:11434", "http://example.com", "http://user@localhost:11434",
                    "http://localhost:11434/path"):
            with self.assertRaises(ValueError):
                QwenSpatialPlanParser(base_url=url)
        with self.assertRaises(ValueError):
            QwenSpatialPlanParser(timeout=0)

    def test_oversize_input_is_rejected_before_http(self):
        with patch("urllib.request.build_opener") as build, self.assertRaises(SpatialParseError):
            QwenSpatialPlanParser()("x" * 2501)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
