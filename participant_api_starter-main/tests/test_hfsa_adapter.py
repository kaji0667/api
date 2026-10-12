"""Synthetic route/predictor fixtures verify dispatch, contracts and privacy."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfsa_adapter import AdapterFailure, HFSAAdapter, parse_question
from ollama_router import RouterError, route_question as real_route_question


def request(question="Synthetic question", kind="short_text", **kwargs):
    constraint = {"type": kind, **kwargs.pop("constraint", {})}
    return {"question": question, "response_constraint": constraint, **kwargs}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.image = Path("synthetic.png")
        self.classification = Mock()
        self.classification.class_names = ["airport", "windmill", "bridge"]
        self.classification.predict.return_value = {"predictions": [
            {"class_name": "bridge", "probability": .7},
            {"class_name": "windmill", "probability": .2},
            {"class_name": "airport", "probability": .1},
        ]}
        self.detection = Mock()
        self.detection.predict.return_value = SimpleNamespace(
            detection_count=2, original_size=(40, 80),
            boxes=np.array([[1., 2., 4., 6.], [30.2, 20.8, 70.1, 35.2]]),
            scores=np.array([.2, .8]),
            grounding_index=1,
        )
        self.counting = Mock()
        self.counting.predict.return_value = SimpleNamespace(count=3)
        mask = np.zeros((40, 80), dtype=bool)
        mask[10:16, 20:24] = True
        self.refseg = Mock()
        self.refseg.predict.return_value = SimpleNamespace(mask=mask)
        self.vqa = Mock()
        self.vqa.predict.return_value = {"status": "ok", "answer": "ship", "question_type": "category"}
        self.adapter = HFSAAdapter(
            self.classification, self.detection, self.counting,
            self.refseg, vqa=self.vqa, change=None,
        )
        self.route = {"task": "counting"}
        self.router_patch = patch(
            "hfsa_adapter.route_question", side_effect=lambda value: self.route
        )
        self.router = self.router_patch.start()
        self.addCleanup(self.router_patch.stop)

    def select(self, task):
        self.route = {"task": task}

    def predict(self, value, images=None):
        return self.adapter.predict(
            value, [self.image] if images is None else images
        )

    def use_grounding_backend(self, backend):
        self.adapter = HFSAAdapter(
            self.classification, self.detection, self.counting,
            self.refseg, grounding_backend=backend, vqa=self.vqa, change=None,
        )

    def assert_no_predictors_called(self):
        for predictor in (
            self.classification, self.detection, self.counting, self.refseg, self.vqa,
        ):
            predictor.predict.assert_not_called()

    def test_original_request_and_question_are_preserved(self):
        value = request(
            "How many ships are near the bridge?\nOnly answer a number.",
            "integer", item_id="TEST_ROUTE",
        )
        plan = parse_question(value)
        self.router.assert_called_once_with(value)
        self.assertEqual(vars(plan), {
            "task": "counting", "original_question": value["question"],
        })

    def test_scene_choice_uses_probabilities_and_returns_original_key(self):
        self.select("classification")
        value = request(
            kind="single_choice", choices={"A": "机场", "B": "风车"},
            constraint={"values": ["A", "B"]},
        )
        self.assertEqual(self.predict(value), "B")
        self.classification.predict.assert_called_once_with(self.image, topk=3)
        self.detection.predict.assert_not_called()
        self.counting.predict.assert_not_called()
        self.refseg.predict.assert_not_called()

    def test_scene_enum_returns_exact_allowed_alias(self):
        self.select("classification")
        value = request(
            kind="enum", constraint={"values": ["wind turbine", "桥梁"]},
        )
        self.assertEqual(self.predict(value), "桥梁")

    def test_scene_short_text_returns_actual_top_class(self):
        self.select("classification")
        self.assertEqual(self.predict(request()), "bridge")
        self.classification.predict.assert_called_once_with(self.image, topk=1)

    def test_vqa_short_text_calls_question_conditioned_model_verbatim(self):
        self.select("vqa")
        value = request(" \tWhat is shown in this image?\nOnly answer briefly.\r\n ")
        self.assertEqual(self.predict(value), "ship")
        self.vqa.predict.assert_called_once_with(self.image, value["question"])
        for predictor in (self.classification, self.detection, self.counting, self.refseg):
            predictor.predict.assert_not_called()

    def test_vqa_choice_and_enum_preserve_original_values(self):
        self.select("vqa")
        cases = (
            (request("What color is the roof?", "single_choice",
                     choices={"roof-blue": "Blue", "roof-red": "Red", "excluded": "not in vocabulary"},
                     constraint={"values": ["roof-blue", "roof-red"]}), "roof-red",
             {"choices": {"roof-blue": "Blue", "roof-red": "Red"}}),
            (request("What color is it?", "enum", constraint={"values": ["Blue", "Red"]}), "Red",
             {"allowed_answers": ["Blue", "Red"]}),
        )
        for value, expected, kwargs in cases:
            with self.subTest(kind=value["response_constraint"]["type"]):
                self.vqa.predict.reset_mock()
                self.vqa.predict.return_value = {"status": "ok", "answer": expected, "question_type": "color"}
                self.assertEqual(self.predict(value), expected)
                self.vqa.predict.assert_called_once_with(self.image, value["question"], **kwargs)
        self.classification.predict.assert_not_called()

    def test_vqa_partial_choices_return_actual_decoder_original_key(self):
        import torch
        from tasks.vqa.config import SUPPORTED, TYPES
        from tasks.vqa.inference import decode_prediction

        self.select("vqa")
        value = request("What color is the roof?", "single_choice",
                        choices={"roof-gray": "gray", "roof-green": "green", "unknown": "unseen shade"},
                        constraint={"values": ["roof-gray", "roof-green", "unknown"]})
        labels = {"types": list(TYPES), "answers": {t: ["gray", "green"] for t in SUPPORTED},
                  "aliases": {t: {} for t in SUPPORTED}}
        type_logits = torch.full((len(TYPES),), -9.)
        type_logits[TYPES.index("color")] = 9.
        answer_logits = {t: torch.tensor([0., 3.]) for t in SUPPORTED}
        self.vqa.predict.side_effect = lambda image, question, **kwargs: decode_prediction(
            type_logits, answer_logits, labels, question, **kwargs)
        self.assertEqual(self.predict(value), "roof-green")
        self.vqa.predict.assert_called_once_with(self.image, value["question"], choices=value["choices"])
        self.classification.predict.assert_not_called()

    def test_vqa_unmatched_options_fail_without_classification_fallback(self):
        self.select("vqa")
        value = request("What color is the roof?", "single_choice", choices={"A": "red", "B": "blue"},
                        constraint={"values": ["A", "B"]})
        self.vqa.predict.return_value = {"status": "unsupported", "reason": "choice_unmatched"}
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(value)
        self.assertEqual(raised.exception.code, "vqa_choice_unmatched")
        self.classification.predict.assert_not_called()

    def test_vqa_explicit_failures_are_safe_and_do_not_fall_back(self):
        self.select("vqa")
        for reason in ("unsupported_task", "unsupported_language", "text_context_exceeded", "nonfinite_prediction"):
            with self.subTest(reason=reason):
                self.vqa.predict.return_value = {"status": "unsupported", "reason": reason}
                with self.assertRaises(AdapterFailure) as raised:
                    self.predict(request())
                self.assertEqual(raised.exception.code, "vqa_" + reason)
        self.classification.predict.assert_not_called()

    def test_vqa_missing_executor_never_calls_classification(self):
        self.select("vqa")
        self.adapter.vqa = None
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request())
        self.assertEqual(raised.exception.code, "vqa_not_configured")
        self.assert_no_predictors_called()

    def test_vqa_checks_actual_answer_and_allowed_contract(self):
        self.select("vqa")
        for result in (None, {"status": "ok", "answer": None, "question_type": "color"},
                       {"status": "ok", "answer": "left", "question_type": "position"},
                       {"status": "unsupported", "reason": "private question content"}):
            with self.subTest(result=result), self.assertRaises(AdapterFailure) as raised:
                self.vqa.predict.return_value = result
                self.predict(request())
            self.assertEqual(raised.exception.code, "invalid_vqa_result")
        self.vqa.predict.return_value = {"status": "ok", "answer": "red", "question_type": "color"}
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request(kind="enum", constraint={"values": ["Red", "Blue"]}))
        self.assertEqual(raised.exception.code, "vqa_answer_contract")
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request(constraint={"min_length": 10}))
        self.assertEqual(raised.exception.code, "vqa_answer_contract")

    def test_vqa_ambiguous_or_invalid_options_are_not_guessed(self):
        self.select("vqa")
        for values in (["A", "A"], ["missing"], []):
            with self.subTest(values=values), self.assertRaises(AdapterFailure) as raised:
                self.predict(request(kind="single_choice", choices={"A": "red"}, constraint={"values": values}))
            self.assertEqual(raised.exception.code, "vqa_invalid_options")
        self.vqa.predict.assert_not_called()
        self.vqa.predict.side_effect = ValueError("Ambiguous normalized choices: private content")
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request(kind="enum", constraint={"values": ["red", "Red"]}))
        self.assertEqual(raised.exception.code, "vqa_invalid_input")

    def test_scene_unknown_or_duplicate_options_fail_before_inference(self):
        self.select("classification")
        for values in (["windmill", "forest"], ["windmill", "风车"]):
            with self.subTest(values=values), self.assertRaises(AdapterFailure):
                self.predict(request(kind="enum", constraint={"values": values}))
        self.classification.predict.assert_not_called()

    def test_count_integer_uses_independent_counting_predictor(self):
        self.select("counting")
        value = request("How many airplanes are in the image? Only answer a number.", "integer")
        self.assertEqual(self.predict(value), "3")
        self.counting.predict.assert_called_once_with(self.image, value["question"])
        self.detection.predict.assert_not_called()
        self.refseg.predict.assert_not_called()
        self.classification.predict.assert_not_called()

    def test_count_qualified_question_and_whitespace_reach_predictor_verbatim(self):
        self.select("counting")
        value = request(
            " \tHow many ships are in the lower left, except those near the bridge?\n"
            "只回答数字。 Do not count red ships.\r\n ",
            "integer",
        )
        self.assertEqual(self.predict(value), "3")
        self.counting.predict.assert_called_once_with(self.image, value["question"])
        self.assertEqual(parse_question(value).original_question, value["question"])

    def test_count_options_map_prediction_without_clamping(self):
        value = request(
            kind="single_choice", choices={"A": "2", "B": "3"},
            constraint={"values": ["A", "B"]},
        )
        self.assertEqual(self.predict(value), "B")
        with self.assertRaises(AdapterFailure):
            self.predict(request(kind="integer", constraint={"minimum": 5}))
        with self.assertRaises(AdapterFailure):
            self.predict(request(kind="enum", constraint={"values": ["1", "2"]}))

    def test_presence_uses_detection_with_complete_question(self):
        self.select("presence")
        value = request(
            " \t图中有没有红色屋顶的房屋？\nOnly answer Yes or No. \n",
            "enum", constraint={"values": ["No", "Yes"]},
        )
        self.assertEqual(self.predict(value), "Yes")
        self.detection.predict.assert_called_once_with(
            self.image, value["question"], require_grounding=False
        )
        self.counting.predict.assert_not_called()
        self.refseg.predict.assert_not_called()

    def test_presence_negative_returns_original_choice_key(self):
        self.select("presence")
        self.detection.predict.return_value.detection_count = 0
        value = request(
            kind="single_choice", choices={"A": "是", "B": "否"},
            constraint={"values": ["A", "B"]},
        )
        self.assertEqual(self.predict(value), "B")

    def test_presence_rejects_non_boolean_answer_options(self):
        self.select("presence")
        with self.assertRaises(AdapterFailure):
            self.predict(request(
                kind="enum", constraint={"values": ["red", "blue"]},
            ))

    def test_visual_grounding_uses_description_assignment_in_original_pixels(self):
        question = (
            "Locate the red warehouse at the lower left. To its right is a road."
        )
        self.select("visual_grounding")
        value = request(question, "bbox", image_width=80, image_height=40)
        self.assertEqual(self.predict(value), [30, 20, 71, 36])
        self.detection.predict.assert_called_once_with(self.image, question)
        self.refseg.predict.assert_not_called()
        self.counting.predict.assert_not_called()

    def test_grounding_assignment_beats_opposite_confidence_order(self):
        self.select("visual_grounding")
        self.detection.predict.return_value.grounding_index = 0
        self.assertEqual(self.predict(request(kind="bbox", image_width=80, image_height=40)), [1, 2, 4, 6])

    def test_missing_or_invalid_match_never_falls_back_to_confidence(self):
        self.select("visual_grounding")
        for index in (None, True, -1, 2, 0.0):
            self.detection.predict.return_value.grounding_index = index
            with self.subTest(index=index), self.assertRaises(AdapterFailure) as context:
                self.predict(request(kind="bbox", image_width=80, image_height=40))
            self.assertEqual(context.exception.code, "invalid_grounding_match")

    def test_count_ranges_map_uniquely_without_clamping(self):
        value = request(kind="single_choice", choices={"A": "0-2", "B": "3 or more"},
                        constraint={"values": ["A", "B"]})
        self.assertEqual(self.predict(value), "B")
        value["choices"]["A"] = "0-3"
        with self.assertRaises(AdapterFailure):
            self.predict(value)

    def test_bbox_wrapper_and_answer_instructions_reach_detection_verbatim(self):
        description = "The red warehouse at the lower left. A road is above it. Exclude the white building."
        self.select("visual_grounding")
        value = request(
            " \tReturn a pixel box only.\nDescription: " + description + "\nOnly output [x1,y1,x2,y2]. \n",
            "bbox", image_width=80, image_height=40,
        )
        self.assertEqual(self.predict(value), [30, 20, 71, 36])
        self.detection.predict.assert_called_once_with(self.image, value["question"])
        self.refseg.predict.assert_not_called()
        self.assertEqual(parse_question(value).original_question, value["question"])

    def test_real_bbox_route_reaches_detection_without_ollama(self):
        question = " \tLocate it.\nDescription: The lawn at the lower left. A road is above it. \n"
        value = request(question, "bbox", images=[{}], image_width=80, image_height=40)
        self.detection.predict.return_value.grounding_index = 0
        with patch("hfsa_adapter.route_question", wraps=real_route_question), \
                patch("ollama_router.urllib.request.build_opener") as opener:
            self.assertEqual(self.predict(value), [1, 2, 4, 6])
            self.assertEqual(parse_question(value).original_question, question)
            opener.assert_not_called()
        self.detection.predict.assert_called_once_with(self.image, question)
        self.refseg.predict.assert_not_called()
        self.counting.predict.assert_not_called()
        self.vqa.predict.assert_not_called()

    def test_real_bbox_route_rejects_two_images_before_visual_inference(self):
        value = request(kind="bbox", images=[{}, {}], image_width=80, image_height=40)
        with patch("hfsa_adapter.route_question", wraps=real_route_question), \
                patch("ollama_router.urllib.request.build_opener") as opener:
            with self.assertRaises(AdapterFailure) as context:
                self.predict(value, images=[self.image, self.image])
            self.assertEqual(context.exception.code, "router_no_compatible_executor")
            opener.assert_not_called()
        self.assert_no_predictors_called()

    def test_detection_bbox_rounds_outward_and_clips_all_edges(self):
        self.select("visual_grounding")
        self.detection.predict.return_value.boxes = np.array([
            [5., 5., 8., 8.], [-2.2, -1.4, 83.1, 42.6],
        ])
        self.assertEqual(
            self.predict(request(kind="bbox", image_width=80, image_height=40)),
            [0, 0, 80, 40],
        )
        self.refseg.predict.assert_not_called()

    def test_empty_detections_fail_without_refseg_fallback(self):
        self.select("visual_grounding")
        self.detection.predict.return_value = SimpleNamespace(
            detection_count=0, original_size=(40, 80),
            boxes=np.empty((0, 4)), scores=np.empty((0,)),
        )
        with self.assertRaises(AdapterFailure) as context:
            self.predict(request(kind="bbox", image_width=80, image_height=40))
        self.assertEqual(context.exception.code, "empty_detections")
        self.refseg.predict.assert_not_called()

    def test_detection_original_dimensions_must_match_request(self):
        self.select("visual_grounding")
        with self.assertRaises(AdapterFailure) as context:
            self.predict(request(kind="bbox", image_width=40, image_height=40))
        self.assertEqual(context.exception.code, "image_dimensions_mismatch")
        self.refseg.predict.assert_not_called()

    def test_invalid_detection_shape_count_or_nonfinite_values_are_rejected(self):
        self.select("visual_grounding")
        cases = (
            {"boxes": np.ones((2, 3))},
            {"scores": np.ones((2, 1))},
            {"scores": np.ones((1,))},
            {"detection_count": 1},
            {"boxes": np.array([[float("nan"), 2., 4., 6.], [30.2, 20.8, 70.1, 35.2]])},
            {"boxes": np.array([[1., 2., 4., 6.], [30.2, 20.8, float("inf"), 35.2]])},
            {"scores": np.array([.2, float("nan")])},
        )
        for index, changed in enumerate(cases):
            fields = {
                "detection_count": 2, "original_size": (40, 80),
                "boxes": np.array([[1., 2., 4., 6.], [30.2, 20.8, 70.1, 35.2]]),
                "scores": np.array([.2, .8]),
                **changed,
            }
            self.detection.predict.return_value = SimpleNamespace(**fields)
            with self.subTest(case=index), self.assertRaises(AdapterFailure) as context:
                self.predict(request(kind="bbox", image_width=80, image_height=40))
            self.assertEqual(context.exception.code, "invalid_detection_result")
        self.refseg.predict.assert_not_called()

    def test_selected_detection_empty_after_clipping_is_rejected(self):
        self.select("visual_grounding")
        self.detection.predict.return_value.boxes = np.array([
            [1., 2., 4., 6.], [81.2, 2., 85.3, 6.],
        ])
        with self.assertRaises(AdapterFailure) as context:
            self.predict(request(kind="bbox", image_width=80, image_height=40))
        self.assertEqual(context.exception.code, "empty_detection_box")
        self.refseg.predict.assert_not_called()

    def test_zero_width_fractional_box_is_rejected_before_integer_rounding(self):
        self.select("visual_grounding")
        self.detection.predict.return_value.boxes = np.array([
            [1., 2., 4., 6.], [30.2, 20.8, 30.2, 35.2],
        ])
        with self.assertRaises(AdapterFailure) as context:
            self.predict(request(kind="bbox", image_width=80, image_height=40))
        self.assertEqual(context.exception.code, "empty_detection_box")

    def test_invalid_grounding_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            self.use_grounding_backend("unknown")

    def test_explicit_refseg_backend_keeps_mask_box_path(self):
        self.use_grounding_backend("refseg")
        self.select("visual_grounding")
        value = request("Locate the gray small windmill. Only return a box.", "bbox",
                        image_width=80, image_height=40)
        self.assertEqual(
            self.predict(value),
            [20, 10, 24, 16],
        )
        self.refseg.predict.assert_called_once_with(self.image, value["question"])
        self.detection.predict.assert_not_called()

    def test_refseg_receives_multiline_mixed_case_question_verbatim(self):
        self.use_grounding_backend("refseg")
        description = "The SMALL warehouse.\nA river lies to its left; do not include the bridge."
        self.select("visual_grounding")
        value = request(" \tLocate it. dEsCrIpTiOn：\n" + description + "\n只返回框。 \n",
                        "bbox", image_width=80, image_height=40)
        self.assertEqual(self.predict(value), [20, 10, 24, 16])
        self.refseg.predict.assert_called_once_with(self.image, value["question"])
        self.assertEqual(parse_question(value).original_question, value["question"])

    def test_description_label_does_not_choose_a_task(self):
        self.select("vqa")
        value = request("Description: The red warehouse. What color is it?", "short_text")
        self.assertEqual(parse_question(value).task, "vqa")
        self.assertEqual(self.predict(value), "ship")
        self.vqa.predict.assert_called_once_with(self.image, value["question"])
        self.classification.predict.assert_not_called()
        self.detection.predict.assert_not_called()

    def test_empty_description_wrapper_is_forwarded_without_target_extraction(self):
        self.select("visual_grounding")
        value = request("Locate it. Description:  \n", "bbox", image_width=80, image_height=40)
        self.assertEqual(self.predict(value), [30, 20, 71, 36])
        self.detection.predict.assert_called_once_with(self.image, value["question"])

    def test_unwrapped_grounding_keeps_entire_source_question(self):
        self.select("visual_grounding")
        value = request("Locate the warehouse above the river. Return only a box.", "bbox")
        self.assertEqual(parse_question(value).original_question, value["question"])

    def test_chinese_description_and_question_are_forwarded_without_translation(self):
        self.select("visual_grounding")
        value = request(" \t定位目标。\nDescription: 道路上方的红色仓库\n只返回像素框。 \n",
                        "bbox", image_width=80, image_height=40)
        for backend, predictor, box in (
            ("detection", self.detection, [30, 20, 71, 36]),
            ("refseg", self.refseg, [20, 10, 24, 16]),
        ):
            with self.subTest(backend=backend):
                self.use_grounding_backend(backend)
                self.assertEqual(self.predict(value), box)
                predictor.predict.assert_called_once_with(self.image, value["question"])
                predictor.reset_mock()

    def test_router_oversize_failure_stops_before_predictor(self):
        self.router.side_effect = RouterError("router_input_too_long")
        with self.assertRaises(AdapterFailure) as context:
            self.predict(request("Locate it. Description: " + "x" * 2501, "bbox",
                                 image_width=80, image_height=40))
        self.assertEqual(context.exception.code, "router_input_too_long")
        self.assert_no_predictors_called()

    def test_empty_refseg_mask_fails(self):
        self.use_grounding_backend("refseg")
        self.select("visual_grounding")
        self.refseg.predict.return_value.mask[:] = False
        with self.assertRaises(AdapterFailure):
            self.predict(request(kind="bbox", image_width=80, image_height=40))

    def test_refseg_dimension_mismatch_fails(self):
        self.use_grounding_backend("refseg")
        self.select("visual_grounding")
        with self.assertRaises(AdapterFailure):
            self.predict(request(kind="bbox", image_width=40, image_height=40))

    def test_unsupported_tasks_never_fall_through_to_detection(self):
        self.select("unrecognized_task")
        with self.assertRaises(AdapterFailure):
            self.predict(request())
        self.assert_no_predictors_called()

    def test_empty_source_question_fails_before_predictor(self):
        self.select("counting")
        for question in ("", " \t\r\n "):
            with self.subTest(question=question), self.assertRaises(AdapterFailure) as context:
                self.predict(request(question, "integer"))
            self.assertEqual(context.exception.code, "router_empty_question")
        self.assert_no_predictors_called()

    def test_extra_router_fields_fail_before_predictor(self):
        for field, value in (("target", "ships"), ("conditions", []), ("other", True)):
            self.route = {"task": "counting", field: value}
            with self.subTest(field=field), self.assertRaises(AdapterFailure) as context:
                self.predict(request(kind="integer"))
            self.assertEqual(context.exception.code, "router_invalid_output")
        self.assert_no_predictors_called()

    def test_task_answer_contract_mismatch_fails_before_predictor(self):
        for task, kind in (
            ("classification", "integer"),
            ("presence", "bbox"),
            ("counting", "bbox"),
            ("visual_grounding", "short_text"),
            ("vqa", "integer"),
            ("vqa", "bbox"),
        ):
            self.select(task)
            with self.subTest(task=task, kind=kind), self.assertRaises(AdapterFailure):
                self.predict(request(kind=kind))
        self.assert_no_predictors_called()

    def test_two_images_never_run_single_image_predictors(self):
        self.select("change")
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request(), [self.image, self.image])
        self.assertEqual(raised.exception.code, "change_not_configured")
        self.select("counting")
        with self.assertRaises(AdapterFailure):
            self.predict(request(kind="integer"), [self.image, self.image])
        self.select("vqa")
        with self.assertRaises(AdapterFailure) as raised:
            self.predict(request(), [self.image, self.image])
        self.assertEqual(raised.exception.code, "unsupported_image_count")
        self.assert_no_predictors_called()


class DiagnosticTests(unittest.TestCase):
    setUp = AdapterTests.setUp
    select = AdapterTests.select
    predict = AdapterTests.predict
    assert_no_predictors_called = AdapterTests.assert_no_predictors_called

    def test_format_conflict_logs_validated_task_without_target(self):
        self.select("visual_grounding")
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(AdapterFailure):
            self.predict(request("private-description private-key", "short_text"))
        record = json.loads(output.getvalue())
        self.assertEqual((record["reason"], record["task"]),
                         ("task_constraint_mismatch", "visual_grounding"))
        self.assertNotIn("private-description", output.getvalue())
        self.assertNotIn("private-key", output.getvalue())
        self.assert_no_predictors_called()

    def test_router_format_conflict_preserves_safe_task_in_diagnostic(self):
        self.router.side_effect = RouterError("task_constraint_mismatch", task="visual_grounding")
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(AdapterFailure):
            self.predict(request(kind="single_choice"))
        record = json.loads(output.getvalue())
        self.assertEqual((record["reason"], record["task"]),
                         ("task_constraint_mismatch", "visual_grounding"))
        self.assert_no_predictors_called()

    def test_exception_task_cannot_log_unvalidated_text(self):
        self.router.side_effect = RouterError("router_invalid_output", task="private-description")
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(AdapterFailure):
            self.predict(request())
        self.assertIsNone(json.loads(output.getvalue())["task"])
        self.assertNotIn("private-description", output.getvalue())

    def test_other_exception_task_attribute_is_not_logged(self):
        error = RuntimeError("private-message")
        error.task = "private-description"
        self.router.side_effect = error
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(RuntimeError):
            self.predict(request())
        record = json.loads(output.getvalue())
        self.assertEqual(record["reason"], "unexpected_exception")
        self.assertIsNone(record["task"])
        self.assertNotIn("private-description", output.getvalue())
        self.assertNotIn("private-message", output.getvalue())

    def test_router_failure_remains_private(self):
        self.router.side_effect = AdapterFailure(
            "router_invalid_output", "private question private-key"
        )
        output = io.StringIO()
        value = request(
            "Describe this private question text.",
            item_id="TEST_DIAGNOSTIC", request_id="test:diagnostic:1",
        )
        with redirect_stderr(output), self.assertRaises(AdapterFailure):
            self.predict(value)
        record = json.loads(output.getvalue())
        self.assertEqual(record["reason"], "router_invalid_output")
        self.assertEqual(record["item_id"], "TEST_DIAGNOSTIC")
        self.assertIsNone(record["task"])
        self.assertNotIn("private question", output.getvalue())
        self.assertNotIn("private-key", output.getvalue())
        self.assert_no_predictors_called()

    def test_unexpected_exception_message_and_paths_remain_private(self):
        self.select("classification")
        self.classification.predict.side_effect = RuntimeError(
            "private-prompt private-key /private/model/path"
        )
        output = io.StringIO()
        value = request(item_id="TEST_DIAGNOSTIC")
        with redirect_stderr(output), self.assertRaises(RuntimeError):
            self.predict(value)
        record = json.loads(output.getvalue())
        self.assertEqual(
            (record["task"], record["reason"], record["error_type"]),
            ("classification", "unexpected_exception", "RuntimeError"),
        )
        for secret in ("private-prompt", "private-key", "/private/model/path"):
            self.assertNotIn(secret, output.getvalue())

    def test_success_log_contains_task_and_no_answer(self):
        output = io.StringIO()
        value = request(kind="integer", item_id="TEST_DIAGNOSTIC")
        with redirect_stderr(output):
            self.assertEqual(self.predict(value), "3")
        record = json.loads(output.getvalue())
        self.assertEqual((record["status"], record["task"]), ("succeeded", "counting"))
        self.assertNotIn("answer", record)

    def test_count_diagnostic_keeps_question_and_answer_private_without_scope_flags(self):
        value = request("How many ships are within private-area?", kind="integer")
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertEqual(self.predict(value), "3")
        self.counting.predict.assert_called_once_with(self.image, value["question"])
        record = json.loads(output.getvalue())
        self.assertNotIn("counting_scope", record)
        self.assertNotIn("counting_conditions_ignored", record)
        self.assertNotIn("private-area", output.getvalue())
        self.assertNotIn("answer", record)

    def test_vqa_log_identifies_actual_vqa_executor(self):
        self.select("vqa")
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertEqual(self.predict(request("private-question")), "ship")
        record = json.loads(output.getvalue())
        self.assertEqual((record["task"], record["executor"], record["status"]),
                         ("vqa", "vqa", "succeeded"))
        self.assertNotIn("private-question", output.getvalue())
        self.assertNotIn("answer", record)

    def test_empty_grounding_diagnostic_contains_only_allowed_numeric_fields(self):
        self.select("visual_grounding")
        self.detection.predict.return_value = SimpleNamespace(
            detection_count=0, original_size=(40, 80),
            boxes=np.empty((0, 4)), scores=np.empty(0), grounding_index=None,
            diagnostics={"input_hw": [448, 832], "original_hw": [40, 80],
                         "confidence_threshold": .25, "max_confidence": .12,
                         "raw_anchor_count": 200, "pre_nms_above_threshold": 0,
                         "nms_candidate_count": 0, "prompt": "private-description",
                         "answer": "private-answer", "unexpected": "private-key"},
        )
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(AdapterFailure) as raised:
            self.predict(request("private-description", "bbox", image_height=40, image_width=80))
        self.assertEqual(raised.exception.code, "empty_detections")
        record = json.loads(output.getvalue())
        self.assertEqual(record["detection_inference"]["nms_candidate_count"], 0)
        self.assertEqual(record["detection_inference"]["max_confidence"], .12)
        self.assertNotIn("private-", output.getvalue())
        self.assertNotIn("answer", record)

    def test_failed_request_capture_excludes_credentials_and_gold(self):
        self.select("change")
        value = request(
            "Describe this private replay question.",
            protocol_version="2.0", item_id="TEST_REPLAY",
            request_id="test:replay:1",
        )
        value.update(
            headers={"Authorization": "secret-header"},
            api_key="secret-key", answer="secret-gold",
        )
        value["images"] = [{
            "asset_id": "a" * 32, "sha256": "b" * 64,
            "mime_type": "image/png", "path": "/secret-path",
        }]
        value["response_constraint"]["api_key"] = "secret-constraint"
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"HFSA_API_REPLAY_DIR": directory}):
                output = io.StringIO()
                for _ in range(2):
                    with redirect_stderr(output), self.assertRaises(AdapterFailure):
                        self.predict(value)
                files = list(Path(directory).glob("*.json"))
                self.assertEqual(len(files), 1)
                saved = files[0].read_text(encoding="utf-8")
                payload = json.loads(saved)
        self.assertEqual(payload["question"], value["question"])
        self.assertNotIn(value["question"], output.getvalue())
        self.assertEqual(payload["response_constraint"], {"type": "short_text"})
        for secret in (
            "secret-header", "secret-key", "secret-gold",
            "/secret-path", "secret-constraint",
        ):
            self.assertNotIn(secret, saved)

    def test_capture_failure_does_not_change_successful_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "private-path"
            invalid.write_text("occupied", encoding="utf-8")
            output = io.StringIO()
            with patch.dict("os.environ", {"HFSA_API_REPLAY_DIR": str(invalid)}):
                with redirect_stderr(output):
                    self.assertEqual(self.predict(request(kind="integer")), "3")
        rows = list(map(json.loads, output.getvalue().splitlines()))
        self.assertEqual(rows[0]["event"], "hfsa_api_replay_capture")
        self.assertEqual(rows[-1]["status"], "succeeded")
        self.assertNotIn("private-path", output.getvalue())


class LoadingBoundaryTests(unittest.TestCase):
    def fake_predictor_modules(self):
        factories = {name: Mock() for name in (
            "classification", "detection", "counting", "refseg", "encoder", "vqa", "change", "binary_change",
        )}
        modules = {}
        attributes = {
            "native_classification": {
                "NativeClassification": factories["classification"],
            },
            "native_detection": {
                "NativeDetection": factories["detection"],
                "NativeTextEncoder": factories["encoder"],
            },
            "native_counting": {"NativeCounting": factories["counting"]},
            "native_refseg": {"NativeRefSegPredictor": factories["refseg"]},
            "tasks.vqa.inference": {"VQAPredictor": factories["vqa"]},
            "tasks.change.inference": {"ChangePredictor": factories["change"]},
            "tasks.change.binary_change": {"BinaryChangePredictor": factories["binary_change"]},
        }
        for name, fields in attributes.items():
            module = ModuleType(name)
            for key, value in fields.items():
                setattr(module, key, value)
            modules[name] = module
        return modules, factories

    def test_default_detection_loading_needs_no_refseg_weights_or_model_import(self):
        modules, factories = self.fake_predictor_modules()
        real_import = __import__

        def guard_import(name, *args, **kwargs):
            if name in {"native_refseg", "tasks.refseg.inference", "tasks.routing.adapters.refseg",
                        "tasks.classification.inference", "tasks.detection.inference", "tasks.counting.inference"}:
                raise AssertionError("Native factories must not import a Web Predictor")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"HFSA_MODELS_DIR": "/synthetic/models", "HFSA_API_DEVICE": "cpu"}, clear=True):
            with patch.dict(sys.modules, modules), patch("hfsa_adapter.ensure_router_ready") as ready, \
                    patch("hfsa_adapter.validate_api_vqa_bundle"):
                with patch("hfsa_adapter.Path.is_file", autospec=True, return_value=True) as is_file:
                    with patch("builtins.__import__", side_effect=guard_import), redirect_stdout(io.StringIO()):
                        adapter = HFSAAdapter.from_local()
        self.assertEqual(adapter.grounding_backend, "detection")
        self.assertIsNone(adapter.refseg)
        ready.assert_called_once_with()
        factories["refseg"].assert_not_called()
        self.assertFalse(any(call.args[0].name == "best_raw.pt" for call in is_file.call_args_list))
        for name in ("classification", "detection", "counting", "vqa", "change", "binary_change"):
            factories[name].assert_called_once()
        self.assertIs(adapter.vqa, factories["vqa"].return_value)
        self.assertIs(adapter.change, factories["change"].return_value)
        self.assertEqual(factories["change"].call_args.kwargs["bundle_dir"].name, "mci")
        self.assertEqual(factories["change"].call_args.kwargs["device"], "cpu")
        self.assertIs(factories["change"].call_args.kwargs["binary_predictor"], factories["binary_change"].return_value)
        self.assertEqual(factories["binary_change"].call_args.kwargs["bundle_dir"].name, "changeformer_dsifn")
        self.assertEqual(factories["binary_change"].call_args.kwargs["device"], "cpu")
        self.assertTrue(any(call.args[0].name == "manifest.json" and call.args[0].parent.name == "mci"
                            for call in is_file.call_args_list))
        self.assertEqual(factories["vqa"].call_args.kwargs["bundle_dir"].name, "qformer")
        self.assertEqual(factories["vqa"].call_args.kwargs["device"], "cpu")
        self.assertEqual(set(factories["vqa"].call_args.kwargs["asset_paths"]), {"visual_backbone", "text_encoder"})
        factories["encoder"].assert_called_once_with(device="cpu")
        self.assertIs(factories["detection"].call_args.kwargs["encoder"], factories["encoder"].return_value)
        self.assertEqual(factories["detection"].call_args.kwargs["max_image_size"], 1600)
        self.assertIs(factories["counting"].call_args.kwargs["encoder"], factories["encoder"].return_value)

    def test_explicit_refseg_loading_checks_and_constructs_refseg(self):
        modules, factories = self.fake_predictor_modules()
        with patch.dict(os.environ, {
            "HFSA_MODELS_DIR": "/synthetic/models", "HFSA_API_DEVICE": "cpu",
            "HFSA_GROUNDING_BACKEND": "refseg",
        }, clear=True):
            with patch.dict(sys.modules, modules), patch("hfsa_adapter.ensure_router_ready") as ready, \
                    patch("hfsa_adapter.validate_api_vqa_bundle"):
                with patch("hfsa_adapter.Path.is_file", autospec=True, return_value=True) as is_file, redirect_stdout(io.StringIO()):
                    adapter = HFSAAdapter.from_local()
        self.assertEqual(adapter.grounding_backend, "refseg")
        self.assertIs(adapter.refseg, factories["refseg"].return_value)
        ready.assert_called_once_with()
        factories["refseg"].assert_called_once()
        factories["change"].assert_called_once()
        self.assertIs(adapter.change, factories["change"].return_value)
        self.assertTrue(any(call.args[0].name == "best_raw.pt" for call in is_file.call_args_list))

    def test_change_bundle_and_device_overrides_are_passed_to_one_independent_factory(self):
        modules, factories = self.fake_predictor_modules()
        with patch.dict(os.environ, {
            "HFSA_MODELS_DIR": "/synthetic/models", "HFSA_API_DEVICE": "cpu",
            "HFSA_CHANGE_BUNDLE_DIR": "/synthetic/independent-change", "HFSA_CHANGE_DEVICE": "cpu",
            "HFSA_BINARY_CHANGE_BUNDLE_DIR": "/synthetic/independent-binary", "HFSA_BINARY_CHANGE_DEVICE": "cpu",
        }, clear=True):
            with patch.dict(sys.modules, modules), patch("hfsa_adapter.ensure_router_ready"), \
                    patch("hfsa_adapter.validate_api_vqa_bundle"):
                with patch("hfsa_adapter.Path.is_file", autospec=True, return_value=True), \
                        redirect_stdout(io.StringIO()):
                    adapter = HFSAAdapter.from_local()
        factories["change"].assert_called_once_with(
            bundle_dir=Path("/synthetic/independent-change").resolve(), device="cpu",
            binary_predictor=factories["binary_change"].return_value,
        )
        factories["binary_change"].assert_called_once_with(
            bundle_dir=Path("/synthetic/independent-binary").resolve(), device="cpu",
        )
        self.assertIs(adapter.change, factories["change"].return_value)

    def test_missing_change_manifest_fails_before_any_model_construction(self):
        modules, factories = self.fake_predictor_modules()
        with patch.dict(os.environ, {"HFSA_MODELS_DIR": "/synthetic/models"}, clear=True):
            with patch.dict(sys.modules, modules), patch("hfsa_adapter.ensure_router_ready"), \
                    patch("hfsa_adapter.validate_api_vqa_bundle"):
                with patch("hfsa_adapter.Path.is_file", autospec=True,
                           side_effect=lambda path: not (path.name == "manifest.json" and path.parent.name == "mci")), \
                        redirect_stdout(io.StringIO()):
                    with self.assertRaises(FileNotFoundError):
                        HFSAAdapter.from_local()
        for factory in factories.values():
            factory.assert_not_called()

    def test_invalid_change_bundle_stops_before_shared_model_construction(self):
        modules, factories = self.fake_predictor_modules()
        factories["change"].side_effect = ValueError("synthetic invalid change bundle")
        with patch.dict(os.environ, {"HFSA_MODELS_DIR": "/synthetic/models"}, clear=True):
            with patch.dict(sys.modules, modules), patch("hfsa_adapter.ensure_router_ready"), \
                    patch("hfsa_adapter.validate_api_vqa_bundle"):
                with patch("hfsa_adapter.Path.is_file", autospec=True, return_value=True), \
                        redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        HFSAAdapter.from_local()
        factories["change"].assert_called_once()
        for name, factory in factories.items():
            if name not in {"change", "binary_change"}:
                factory.assert_not_called()

    def test_missing_binary_manifest_fails_before_any_model_construction(self):
        modules, factories = self.fake_predictor_modules()
        with patch.dict(os.environ, {"HFSA_MODELS_DIR": "/synthetic/models"}, clear=True):
            with patch.dict(sys.modules, modules), patch("hfsa_adapter.ensure_router_ready"), \
                    patch("hfsa_adapter.validate_api_vqa_bundle"), \
                    patch("hfsa_adapter.Path.is_file", autospec=True,
                          side_effect=lambda path: not (path.name == "manifest.json" and path.parent.name == "changeformer_dsifn")), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    HFSAAdapter.from_local()
        for factory in factories.values():
            factory.assert_not_called()

    def test_invalid_binary_bundle_stops_before_mci_or_shared_model_construction(self):
        modules, factories = self.fake_predictor_modules()
        factories["binary_change"].side_effect = ValueError("synthetic invalid binary bundle")
        with patch.dict(os.environ, {"HFSA_MODELS_DIR": "/synthetic/models"}, clear=True):
            with patch.dict(sys.modules, modules), patch("hfsa_adapter.ensure_router_ready"), \
                    patch("hfsa_adapter.validate_api_vqa_bundle"), \
                    patch("hfsa_adapter.Path.is_file", autospec=True, return_value=True), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    HFSAAdapter.from_local()
        factories["binary_change"].assert_called_once()
        for name, factory in factories.items():
            if name != "binary_change":
                factory.assert_not_called()

    def test_invalid_backend_fails_before_router_or_weight_loading(self):
        with patch.dict(os.environ, {"HFSA_GROUNDING_BACKEND": "unknown"}):
            with patch("hfsa_adapter.ensure_router_ready") as ready, patch("hfsa_adapter.Path.is_file") as is_file:
                with self.assertRaises(ValueError):
                    HFSAAdapter.from_local()
        ready.assert_not_called()
        is_file.assert_not_called()

    def test_invalid_detection_size_fails_before_router_or_weight_loading(self):
        for value in ("640", "801", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {"HFSA_DETECTION_MAX_IMGSZ": value}):
                with patch("hfsa_adapter.ensure_router_ready") as ready, patch("hfsa_adapter.Path.is_file") as is_file:
                    with self.assertRaises(ValueError):
                        HFSAAdapter.from_local()
                ready.assert_not_called()
                is_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
