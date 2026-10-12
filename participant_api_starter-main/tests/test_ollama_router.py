"""Deterministic bbox routing and mocked selector HTTP/failure boundaries."""

import json
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollama_router import RouterError, SCHEMA, TASKS, route_question, schema_for_request


class RouterTests(unittest.TestCase):
    def setUp(self):
        environment = {
            "HFSA_ROUTER_URL": "http://127.0.0.1:11434",
            "HFSA_ROUTER_MODEL": "qwen3:1.7b",
            "HFSA_ROUTER_TIMEOUT": "30",
        }
        self.environment_patch = patch.dict(os.environ, environment)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        self.response = Mock()
        self.response.__enter__ = Mock(return_value=self.response)
        self.response.__exit__ = Mock(return_value=False)
        self.opener = Mock()
        self.opener.open.return_value = self.response
        self.opener_patch = patch(
            "ollama_router.urllib.request.build_opener", return_value=self.opener
        )
        self.build_opener = self.opener_patch.start()
        self.addCleanup(self.opener_patch.stop)
        self.value = {
            "question": "How many ships are there in the image?",
            "response_constraint": {"type": "integer"},
            "images": [{}],
        }
        self.reply({"task": "counting"})

    def reply(self, plan, done=True):
        self.response.read.return_value = json.dumps({
            "done": done, "message": {"content": json.dumps(plan)},
        }).encode("utf-8")

    def sent_body(self):
        call = self.opener.open.call_args.args[0]
        return call, json.loads(call.data)

    def test_success_uses_schema_model_and_disabled_thinking(self):
        self.assertEqual(route_question(self.value), {"task": "counting"})
        call, body = self.sent_body()
        self.assertEqual(call.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(body["model"], "qwen3:1.7b")
        self.assertIs(body["think"], False)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["format"]["properties"]["task"]["enum"],
                         ["counting"])
        self.assertEqual(body["format"]["required"], SCHEMA["required"])
        self.assertEqual(body["format"]["required"], ["task"])
        self.assertEqual(set(body["format"]["properties"]), {"task"})
        self.assertIs(body["format"]["additionalProperties"], False)
        self.assertEqual(SCHEMA["properties"]["task"]["enum"], list(TASKS))
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["num_predict"], 64)
        self.assertEqual(self.opener.open.call_args.kwargs["timeout"], 30)
        self.response.read.assert_called_once_with(65537)
        self.assertEqual(self.build_opener.call_args.args[0].proxies, {})

    def test_invalid_outer_or_plan_json_fails_without_fallback(self):
        for raw in (
            b"not json",
            json.dumps({"done": True, "message": {"content": "not json"}}).encode(),
        ):
            with self.subTest(raw=raw):
                self.response.read.return_value = raw
                self.opener.open.reset_mock()
                with self.assertRaises(RouterError) as context:
                    route_question(self.value)
                self.assertEqual(context.exception.code, "router_invalid_output")
                self.opener.open.assert_called_once()

    def test_unknown_task_is_rejected(self):
        self.reply({"task": "detection"})
        with self.assertRaises(RouterError) as context:
            route_question(self.value)
        self.assertEqual(context.exception.code, "router_invalid_output")
        self.opener.open.assert_called_once()

    def test_target_conditions_and_any_extra_field_are_rejected(self):
        for field, value in (("target", "ships"), ("conditions", []), ("other", True)):
            with self.subTest(field=field):
                self.reply({"task": "counting", field: value})
                self.opener.open.reset_mock()
                with self.assertRaises(RouterError) as context:
                    route_question(self.value)
                self.assertEqual(context.exception.code, "router_invalid_output")
                self.opener.open.assert_called_once()

    def test_missing_non_string_and_non_object_task_output_is_rejected(self):
        for plan in ({}, {"task": None}, {"task": ["counting"]}, "counting", []):
            with self.subTest(plan=plan):
                self.reply(plan)
                with self.assertRaises(RouterError) as context:
                    route_question(self.value)
                self.assertEqual(context.exception.code, "router_invalid_output")

    def test_timeout_uses_fixed_error_without_exception_text(self):
        self.opener.open.side_effect = TimeoutError("private URL and prompt")
        with self.assertRaises(RouterError) as context:
            route_question(self.value)
        self.assertEqual(context.exception.code, "router_unavailable_or_timeout")
        self.assertNotIn("private", str(context.exception))
        self.opener.open.assert_called_once()

    def test_non_loopback_url_is_rejected_before_http(self):
        with patch.dict(os.environ, {"HFSA_ROUTER_URL": "http://example.com:11434"}):
            with self.assertRaises(RouterError) as context:
                route_question(self.value)
        self.assertEqual(context.exception.code, "router_requires_loopback")
        self.build_opener.assert_not_called()
        self.opener.open.assert_not_called()

    def test_verbatim_question_choices_and_constraints_reach_task_selector(self):
        value = {
            "question": " \t图中属于哪种场景？\nDescription: The SMALL warehouse.\r\n"
                        "Return one option. Do not delete this sentence. \n",
            "choices": {"A": "airport", "B": "harbor"},
            "response_constraint": {
                "type": "single_choice", "values": ["A", "B"],
                "case_insensitive": False,
            },
            "images": [{}, {}],
        }
        self.reply({"task": "vqa"})
        route_question(value)
        _, body = self.sent_body()
        sent = json.loads(body["messages"][1]["content"])
        for field in ("question", "choices", "response_constraint"):
            self.assertEqual(sent[field], value[field])
        self.assertEqual(sent["image_count"], 2)

    def test_oversize_question_fails_before_http_without_truncation(self):
        value = {**self.value, "question": " \tDescription: " + "红 ships\n" * 400 + " \n"}
        original = value["question"]
        with self.assertRaises(RouterError) as context:
            route_question(value)
        self.assertEqual(context.exception.code, "router_input_too_long")
        self.assertEqual(value["question"], original)
        self.build_opener.assert_not_called()
        self.opener.open.assert_not_called()

    def test_empty_question_fails_before_http(self):
        for question in ("", " \t\r\n ", None):
            with self.subTest(question=question), self.assertRaises(RouterError) as context:
                route_question({**self.value, "question": question})
            self.assertEqual(context.exception.code, "router_empty_question")
        self.build_opener.assert_not_called()
        self.opener.open.assert_not_called()

    def test_non_bbox_formats_exclude_and_reject_bbox_task(self):
        for kind in ("short_text", "single_choice", "enum", "integer"):
            with self.subTest(kind=kind):
                value = {**self.value, "response_constraint": {"type": kind}}
                self.reply({"task": "visual_grounding"})
                with self.assertRaises(RouterError) as context:
                    route_question(value)
                self.assertEqual(context.exception.code, "task_constraint_mismatch")
                self.assertEqual(context.exception.task, "visual_grounding")
                _, body = self.sent_body()
                self.assertNotIn("visual_grounding", body["format"]["properties"]["task"]["enum"])

    def test_bbox_routes_without_http_and_preserves_request(self):
        value = {
            **self.value,
            "question": " \tFind the box.\nDescription: The lawn. A road is above it. \n",
            "response_constraint": {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
            "answer_type": "bbox", "image_width": 80, "image_height": 40,
        }
        original = deepcopy(value)
        self.assertEqual(route_question(value), {"task": "visual_grounding"})
        self.assertEqual(value, original)
        self.assertEqual(schema_for_request(value)["properties"]["task"]["enum"],
                         ["visual_grounding"])
        self.build_opener.assert_not_called()
        self.opener.open.assert_not_called()

    def test_bbox_does_not_depend_on_ollama_availability_or_configuration(self):
        value = {**self.value, "response_constraint": {"type": "bbox"}}
        self.opener.open.side_effect = TimeoutError("Ollama unavailable")
        self.reply({"task": "vqa"})
        with patch.dict(os.environ, {"HFSA_ROUTER_URL": "invalid", "HFSA_ROUTER_TIMEOUT": "invalid"}):
            self.assertEqual(route_question(value), {"task": "visual_grounding"})
        self.build_opener.assert_not_called()
        self.opener.open.assert_not_called()

    def test_bbox_keeps_question_length_and_image_count_validation(self):
        value = {**self.value, "response_constraint": {"type": "bbox"}}
        for changes, expected in (
            ({"question": " \t\r\n"}, "router_empty_question"),
            ({"question": "x" * 2501}, "router_input_too_long"),
            ({"images": []}, "router_invalid_image_count"),
            ({"images": [{}, {}]}, "router_no_compatible_executor"),
            ({"images": [{}, {}, {}]}, "router_invalid_image_count"),
        ):
            with self.subTest(changes=changes), self.assertRaises(RouterError) as context:
                route_question({**value, **changes})
            self.assertEqual(context.exception.code, expected)
        self.build_opener.assert_not_called()
        self.opener.open.assert_not_called()

    def test_change_does_not_accept_bbox_or_integer_contract(self):
        for kind in ("bbox", "integer"):
            with self.subTest(kind=kind):
                value = {**self.value, "images": [{}, {}], "response_constraint": {"type": kind}}
                with self.assertRaises(RouterError) as context:
                    route_question(value)
                self.assertEqual(context.exception.code, "router_no_compatible_executor")
        self.opener.open.assert_not_called()

    def test_two_images_allow_only_two_image_semantic_tasks(self):
        value = {**self.value, "images": [{}, {}], "response_constraint": {"type": "short_text"}}
        self.reply({"task": "change"})
        route_question(value)
        _, body = self.sent_body()
        self.assertEqual(body["format"]["properties"]["task"]["enum"], ["change", "vqa"])
        self.reply({"task": "counting"})
        with self.assertRaises(RouterError) as context:
            route_question(value)
        self.assertEqual(context.exception.code, "task_constraint_mismatch")

    def test_request_schemas_do_not_mutate_global_or_other_requests(self):
        bbox = schema_for_request({"response_constraint": {"type": "bbox"}})
        text = schema_for_request({"response_constraint": {"type": "short_text"}})
        bbox["properties"]["task"]["enum"].clear()
        bbox["required"].clear()
        self.assertEqual(set(text["properties"]), {"task"})
        self.assertEqual(text["required"], ["task"])
        self.assertNotIn("visual_grounding", text["properties"]["task"]["enum"])
        self.assertEqual(SCHEMA["properties"]["task"]["enum"], list(TASKS))
        self.assertEqual(set(SCHEMA["properties"]), {"task"})
        self.assertEqual(SCHEMA["required"], ["task"])

    def test_format_does_not_force_every_single_choice_into_scene_classification(self):
        value = {**self.value, "response_constraint": {"type": "single_choice"}}
        for task in ("vqa", "counting", "presence"):
            with self.subTest(task=task):
                self.reply({"task": task})
                self.assertEqual(route_question(value)["task"], task)

    def test_vqa_cannot_request_integer_answers(self):
        self.reply({"task": "vqa"})
        value = {**self.value, "response_constraint": {"type": "integer"}}
        with self.assertRaises(RouterError) as raised:
            route_question(value)
        self.assertEqual(raised.exception.code, "task_constraint_mismatch")

    def test_invalid_constraint_or_image_count_fails_before_http(self):
        for value in (
            {**self.value, "response_constraint": {"type": "unknown"}},
            {**self.value, "images": []},
            {**self.value, "images": [{}, {}, {}]},
        ):
            with self.subTest(value=value), self.assertRaises(RouterError):
                route_question(value)
        self.opener.open.assert_not_called()

    def test_incomplete_generation_is_rejected(self):
        self.reply({"task": "counting"}, done=False)
        with self.assertRaises(RouterError) as context:
            route_question(self.value)
        self.assertEqual(context.exception.code, "router_incomplete_response")
        self.opener.open.assert_called_once()


if __name__ == "__main__":
    unittest.main()
