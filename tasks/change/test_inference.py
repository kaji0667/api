"""Synthetic preprocessing, region geometry and local answerer contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import urllib.error

import cv2
import numpy as np
from PIL import Image
import torch

from .answering import (EvidenceAnswerer, caption_change_polarity, global_change_query,
                       maximum_region_focus, region_answer)
from .inference import ChangeEvidence, load_pretrained_state, read_pair


def upstream_normalize(rgb):
    """Pinned LEVIR_MCI.py channel loop, independent of deployment constants."""
    result = np.asarray(rgb, np.float32).transpose(2, 0, 1).copy()
    means = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
    stds = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
    for channel in range(3):
        result[channel, :, :] -= means[channel]
        result[channel, :, :] /= stds[channel]
    return result[None]


def evidence(mask=None, caption="Several buildings appeared near a road."):
    if mask is None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[180:190, 10:20] = 2
    return ChangeEvidence(caption, mask, 0.0, (256, 256)).summary()


FOUR = ["Upper Left", "Upper Right", "Lower Left", "Lower Right"]
NINE = ["Top Left", "Top Center", "Top Right", "Middle Left", "Center",
        "Middle Right", "Bottom Left", "Bottom Center", "Bottom Right"]


class MaximumRegionQueryTests(unittest.TestCase):
    def test_explicit_english_area_queries_including_public_runner(self):
        queries = (
            ("Comparing the earlier and later images, which quadrant contains the MOST overall road and building change area? Choose one of the four quadrants.", "all"),
            ("Which region has the largest road change area?", "road"),
            ("Which quadrant shows the greatest building change area?", "building"),
            ("Compare the before and after images. Which region contains the maximum area of detected buildings change?", "building"),
            ("Which region has the greatest building and road changed area? Return only the option key.", "all"),
        )
        for question, focus in queries:
            with self.subTest(question=question):
                self.assertEqual(maximum_region_focus(question), focus)

    def test_explicit_chinese_area_queries(self):
        queries = (
            ("对比前后两幅图像，哪个区域的建筑变化面积最大？只回答选项。", "building"),
            ("哪个象限的道路发生变化的面积最多？", "road"),
            ("比较前后影像，哪一个区域的道路和建筑物变化面积最大？", "all"),
            ("建筑与道路变化面积最大的是哪个区域？", "all"),
        )
        for question, focus in queries:
            with self.subTest(question=question):
                self.assertEqual(maximum_region_focus(question), focus)

    def test_minimum_direction_count_unknown_scope_and_extra_qualifiers_abstain(self):
        unsupported = (
            "Which quadrant has the least building change area?",
            "Which quadrant has the largest newly constructed building change area?",
            "Which quadrant has the largest demolished building change area?",
            "Which quadrant has the most buildings?",
            "Which quadrant has the most building changes?",
            "Which quadrant has the largest water change area?",
            "Which quadrant has the largest overall change area?",
            "Which region changed most?",
            "Which region has the largest building change area near the road?",
            "Which region has the largest building change area? Count only new buildings.",
            "Which region does not have the largest building change area?",
            "哪个区域的建筑变化面积最小？",
            "哪个区域的新增建筑变化面积最大？",
            "哪个区域的拆除建筑变化面积最大？",
            "哪个区域的建筑数量最多？",
            "哪个区域变化最大？",
            "哪个区域的建筑变化面积最大？只考虑新建建筑。",
        )
        for question in unsupported:
            with self.subTest(question=question):
                self.assertIsNone(maximum_region_focus(question))


class GlobalChangeQueryTests(unittest.TestCase):
    def test_unqualified_global_queries_and_unknown_qualifiers(self):
        supported = (
            "Do the earlier and later images show any change? Answer Yes or No.",
            "Compared with the earlier image A, has the later image B changed? Select the original option key.",
            "Are there any changes between the before and after images?",
            "前后两幅图像是否发生变化？只回答是或否。",
            "这两张图片有没有变化？",
        )
        unsupported = (
            "Did a road appear between the earlier and later images?",
            "Is there any change in the upper left of the two images?",
            "Do the earlier and later images show any building change? Answer Yes or No.",
            "Do the earlier and later images show any change? Consider only buildings.",
            "Do the earlier and later images not show any change?",
            "Are the earlier and later images unchanged?",
            "前后两幅图像中的建筑是否发生变化？",
            "前后两幅图像是否没有变化？",
        )
        for question in supported:
            with self.subTest(question=question):
                self.assertTrue(global_change_query(question))
        for question in unsupported:
            with self.subTest(question=question):
                self.assertFalse(global_change_query(question))

    def test_caption_polarity_requires_actual_explicit_predicate(self):
        for caption in ("the scene is the same as before", "There is no change.",
                        "The scene remains unchanged."):
            with self.subTest(caption=caption):
                self.assertIs(caption_change_polarity(caption), False)
        for caption in ("the vegetation has been removed and a road with villas built along appears",
                        "the vegetation has been removed and many houses are built along the road",
                        "A building disappeared."):
            with self.subTest(caption=caption):
                self.assertIs(caption_change_polarity(caption), True)
        for caption in ("Buildings and roads.", "No buildings were added.",
                        "A road might have appeared.", "The scene is the same as before and a road appears.",
                        "12555 building pixels", "Construction is proposed."):
            with self.subTest(caption=caption):
                self.assertIsNone(caption_change_polarity(caption))


class PretrainedStateTests(unittest.TestCase):
    def setUp(self):
        self.module = torch.nn.Linear(3, 2)
        self.state = {key: value.clone() for key, value in self.module.state_dict().items()}
        self.state["released_inactive"] = torch.zeros(4)
        self.ignored = {"released_inactive": {"shape": [4], "dtype": "torch.float32"}}

    def test_exact_audited_subset_loads_every_active_tensor(self):
        self.state["weight"].fill_(7)
        report = load_pretrained_state(self.module, self.state, ignored=self.ignored)
        self.assertTrue(torch.equal(self.module.weight, self.state["weight"]))
        self.assertEqual(report, {"checkpoint_tensors": 3, "matched_tensors": 2,
                                 "ignored_inactive_tensors": 1, "missing": 0,
                                 "unexpected": 0, "strict_active_state": True})

    def test_missing_active_tensor_is_rejected(self):
        del self.state["bias"]
        with self.assertRaises(ValueError):
            load_pretrained_state(self.module, self.state, ignored=self.ignored)

    def test_unlisted_checkpoint_key_is_rejected(self):
        self.state["another_inactive"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            load_pretrained_state(self.module, self.state, ignored=self.ignored)

    def test_absent_audited_key_is_rejected(self):
        del self.state["released_inactive"]
        with self.assertRaises(ValueError):
            load_pretrained_state(self.module, self.state, ignored=self.ignored)

    def test_shape_mismatch_is_rejected_for_active_and_audited_tensors(self):
        for key in ("weight", "released_inactive"):
            with self.subTest(key=key):
                changed = dict(self.state)
                changed[key] = torch.zeros(1)
                with self.assertRaises(ValueError):
                    load_pretrained_state(self.module, changed, ignored=self.ignored)

    def test_dtype_mismatch_is_rejected_for_active_and_audited_tensors(self):
        for key in ("weight", "released_inactive"):
            with self.subTest(key=key):
                changed = dict(self.state)
                changed[key] = changed[key].double()
                with self.assertRaises(ValueError):
                    load_pretrained_state(self.module, changed, ignored=self.ignored)


class ReadPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def save(self, name, array):
        path = self.directory / name
        Image.fromarray(array).save(path)
        return path

    def test_native_256_matches_upstream_channel_normalization_exactly(self):
        rgb = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3).astype(np.uint8)
        other = np.flip(rgb, axis=1).copy()
        a, b, hw = read_pair(self.save("before.png", rgb), self.save("after.png", other))
        self.assertEqual(hw, (256, 256))
        np.testing.assert_array_equal(a.numpy(), upstream_normalize(rgb))
        np.testing.assert_array_equal(b.numpy(), upstream_normalize(other))
        self.assertEqual(a.dtype, torch.float32)
        self.assertTrue(a.is_contiguous())
        self.assertTrue(b.is_contiguous())

    def test_hwc_512_resize_has_correct_shape_and_matches_independent_reference(self):
        rows, columns = np.indices((512, 512))
        rgb = np.stack((rows % 256, columns % 256, (rows + columns) % 256), axis=-1).astype(np.uint8)
        a, b, hw = read_pair(self.save("before.png", rgb), self.save("after.png", 255 - rgb))
        self.assertEqual(hw, (512, 512))
        self.assertEqual(tuple(a.shape), (1, 3, 256, 256))
        self.assertEqual(tuple(b.shape), (1, 3, 256, 256))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.isfinite(b).all())
        np.testing.assert_array_equal(a.numpy(), upstream_normalize(
            cv2.resize(rgb.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)))
        np.testing.assert_array_equal(b.numpy(), upstream_normalize(
            cv2.resize((255 - rgb).astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)))

    def test_rectangular_registered_pair_preserves_original_dimensions(self):
        rgb = np.zeros((128, 512, 3), dtype=np.uint8)
        rgb[:, 256:, 0] = 255
        a, b, hw = read_pair(self.save("before.png", rgb), self.save("after.png", rgb))
        self.assertEqual(hw, (128, 512))
        self.assertEqual(tuple(a.shape), (1, 3, 256, 256))
        self.assertTrue(torch.equal(a, b))
        np.testing.assert_array_equal(a.numpy(), upstream_normalize(
            cv2.resize(rgb.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)))

    def test_different_original_dimensions_fail_instead_of_independent_resize(self):
        a = self.save("before.png", np.zeros((256, 256, 3), dtype=np.uint8))
        b = self.save("after.png", np.zeros((512, 512, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "^image_dimensions_mismatch$"):
            read_pair(a, b)


class RegionAnswerTests(unittest.TestCase):
    def test_each_quadrant_returns_exact_original_enum_value(self):
        for index, expected in enumerate(FOUR):
            with self.subTest(region=expected):
                mask = np.zeros((256, 256), dtype=np.uint8)
                row, column = divmod(index, 2)
                mask[row * 128 + 20:row * 128 + 25, column * 128 + 20:column * 128 + 25] = 2
                result = region_answer({"type": "enum", "values": list(reversed(FOUR))}, None,
                                       evidence(mask), "building")
                self.assertTrue(result.supported)
                self.assertEqual(result.answer, expected)

    def test_each_nine_cell_region_is_unique_and_final_pixel_is_included(self):
        positions = (20, 100, 255)
        for index, expected in enumerate(NINE):
            with self.subTest(region=expected):
                mask = np.zeros((256, 256), dtype=np.uint8)
                row, column = divmod(index, 3)
                mask[positions[row], positions[column]] = 1
                result = region_answer({"type": "enum", "values": NINE}, None, evidence(mask), "road")
                self.assertTrue(result.supported)
                self.assertEqual(result.answer, expected)

    def test_class_focus_uses_area_and_returns_declared_choice_key(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[180:210, 180:210] = 2
        choices = {"q1": "Upper Left", "q2": "Upper Right", "q3": "Lower Left", "q4": "Lower Right"}
        constraint = {"type": "single_choice", "values": list(choices)}
        self.assertEqual(region_answer(constraint, choices, evidence(mask), "road").answer, "q1")
        self.assertEqual(region_answer(constraint, choices, evidence(mask), "building").answer, "q4")
        self.assertEqual(region_answer(constraint, choices, evidence(mask), "all").answer, "q4")

    def test_equal_maximum_area_abstains(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:15, 10:15] = 2
        mask[180:185, 180:185] = 2
        result = region_answer({"type": "enum", "values": FOUR}, None, evidence(mask), "building")
        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "ambiguous_region")

    def test_empty_mask_abstains(self):
        result = region_answer({"type": "enum", "values": FOUR}, None,
                               evidence(np.zeros((256, 256), dtype=np.uint8)), "all")
        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "ambiguous_region")

    def test_unknown_duplicate_semantics_or_incomplete_partition_abstains(self):
        for values in (["Upper Left", "Upper Right", "Lower Left", "unknown region"],
                       ["Upper Left", "northwest", "Lower Left", "Lower Right"],
                       FOUR[:3], FOUR + ["Center"]):
            with self.subTest(values=values):
                result = region_answer({"type": "enum", "values": values}, None, evidence(), "all")
                self.assertFalse(result.supported)
                self.assertEqual(result.reason, "ambiguous_region")


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, limit):
        return self.data[:limit]


class EvidenceAnswererTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {
            "HFSA_ROUTER_URL": "http://127.0.0.1:11434",
            "HFSA_ROUTER_MODEL": "synthetic-model", "HFSA_CHANGE_ANSWER_TIMEOUT": "30",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.answerer = EvidenceAnswerer()
        self.opener = Mock()
        self.opener_patch = patch("tasks.change.answering.urllib.request.build_opener", return_value=self.opener)
        self.opener_patch.start()
        self.addCleanup(self.opener_patch.stop)
        self.valid = {"supported": True, "operation": "answer", "focus": "all",
                      "answer": "buildings", "evidence": ["caption"]}
        self.reply(self.valid)

    def reply(self, value, done=True):
        payload = json.dumps({"done": done, "message": {"content": json.dumps(value)}}).encode()
        self.opener.open.return_value = FakeResponse(payload)

    def answer(self, question="What changed?", constraint=None, choices=None, model_evidence=None):
        return self.answerer.answer(question, constraint or {"type": "short_text"}, choices,
                                    evidence() if model_evidence is None else model_evidence)

    def test_non_loopback_or_credentialed_urls_fail_before_any_http(self):
        for url in ("https://127.0.0.1:11434", "http://example.org:11434", "http://127.0.0.1:11434/api",
                    "http://user:secret@localhost:11434", "http://localhost:11434?x=1", "http://localhost:11434/#part"):
            with self.subTest(url=url), patch.dict("os.environ", {"HFSA_ROUTER_URL": url}):
                with self.assertRaises(ValueError):
                    EvidenceAnswerer()
        self.opener.open.assert_not_called()

    def test_declared_options_original_question_and_schema_reach_mock_ollama(self):
        question = " \tWhich changes occurred?\nOnly select the original key.\r\n "
        constraint = {"type": "single_choice", "values": ["road-key", "building-key"]}
        choices = {"road-key": "road", "building-key": "buildings"}
        self.reply({**self.valid, "answer": "building-key"})
        result = self.answer(question, constraint, choices)
        self.assertTrue(result.supported)
        self.assertEqual(result.answer, "building-key")
        call = self.opener.open.call_args.args[0]
        self.assertEqual(call.full_url, "http://127.0.0.1:11434/api/chat")
        body = json.loads(call.data)
        content = json.loads(body["messages"][1]["content"])
        self.assertEqual(content["question"], question)
        self.assertEqual(content["choices"], choices)
        self.assertEqual(content["response_constraint"], constraint)
        self.assertEqual(body["format"]["properties"]["answer"]["enum"], ["", "road-key", "building-key"])
        self.assertFalse(body["format"]["additionalProperties"])
        self.assertFalse(body["think"])

    def test_caption_answerer_never_receives_mask_statistics_or_other_evidence_fields(self):
        model_evidence = evidence()
        model_evidence["mask_grid"]["building"]["changed_pixels"] = 12555
        model_evidence["untrusted_extra"] = "must not reach the text model"
        result = self.answer(model_evidence=model_evidence)
        self.assertTrue(result.supported)
        body = json.loads(self.opener.open.call_args.args[0].data)
        supplied = json.loads(body["messages"][1]["content"])["model_evidence"]
        self.assertEqual(set(supplied), {"caption", "image_order", "model_scope"})
        self.assertEqual(supplied["caption"], model_evidence["caption"])
        self.assertNotIn("12555", body["messages"][1]["content"])
        self.assertEqual(body["format"]["properties"]["evidence"]["items"]["enum"], ["caption"])

    def test_global_binary_uses_caption_and_preserves_enum_keys_without_llm(self):
        question = "Do the earlier and later images show any change? Answer Yes or No."
        for caption, expected in (("the scene is the same as before", "No"),
                                  ("A road appeared and buildings were built.", "Yes")):
            with self.subTest(caption=caption):
                model_evidence = evidence(caption=caption)
                # Polluting or removing masks cannot change a caption decision.
                model_evidence["mask_grid"] = {"incorrect_pixels": 12555}
                result = self.answer(question, constraint={"type": "enum", "values": ["No", "Yes"]},
                                     model_evidence=model_evidence)
                self.assertEqual((result.supported, result.answer), (True, expected))
        choices = {"changed:key": "Yes", "same-key": "No"}
        result = self.answer("Compared with the earlier image A, has the later image B changed? Select the original option key.",
                             constraint={"type": "single_choice", "values": list(choices)}, choices=choices,
                             model_evidence=evidence(caption="the scene is the same as before"))
        self.assertEqual((result.supported, result.answer), (True, "same-key"))
        self.opener.open.assert_not_called()

    def test_global_binary_short_text_chinese_and_unknown_polarity(self):
        result = self.answer("前后两幅图像是否发生变化？只回答是或否。",
                             model_evidence=evidence(caption="the scene is the same as before"))
        self.assertEqual((result.supported, result.answer), (True, "否"))
        result = self.answer("Do the earlier and later images show any change?",
                             model_evidence=evidence(caption="A building was removed."))
        self.assertEqual((result.supported, result.answer), (True, "Yes"))
        for caption in ("Roads and buildings.", "A road may have appeared.", "No houses were built."):
            with self.subTest(caption=caption):
                result = self.answer("Do the earlier and later images show any change?",
                                     model_evidence=evidence(caption=caption))
                self.assertEqual((result.supported, result.reason), (False, "insufficient_evidence"))
        self.opener.open.assert_not_called()

    def test_binary_options_missing_or_ambiguous_polarity_cannot_force_a_choice(self):
        question = "Do the earlier and later images show any change?"
        for values in (["Yes"], ["No", "False"], ["road", "building"]):
            with self.subTest(values=values):
                result = self.answer(question, constraint={"type": "enum", "values": values},
                                     model_evidence=evidence(caption="the scene is the same as before"))
                self.assertEqual((result.supported, result.reason), (False, "answer_unmatched"))
        self.opener.open.assert_not_called()

    def test_specific_binary_queries_cannot_use_omission_as_no(self):
        questions = (
            "Did a bridge disappear between the images? Answer Yes or No.",
            "Has water flooded the road?",
            "Comparing the images, did a building become red?",
            "两幅图像中的河流是否干涸？",
            "Please determine whether houses were demolished.",
        )
        for question in questions:
            with self.subTest(question=question):
                self.reply({**self.valid, "answer": "No"})
                result = self.answer(question, constraint={"type": "enum", "values": ["Yes", "No"]})
                self.assertEqual((result.supported, result.reason), (False, "unsupported_task"))
        self.opener.open.assert_not_called()

    def test_requested_quantities_are_rejected_before_mask_or_caption_llm(self):
        for question in ("How many individual new buildings were constructed? Return the exact count.",
                         "What is the number of roads that appeared?", "Count the buildings in the later image.",
                         "How much vegetation disappeared?", "新建了多少栋建筑？", "变化建筑的数量是多少？"):
            with self.subTest(question=question):
                self.reply({**self.valid, "answer": "12555"})
                result = self.answer(question)
                self.assertEqual((result.supported, result.reason), (False, "unsupported_task"))
        self.opener.open.assert_not_called()

    def test_numeric_or_boolean_short_caption_answers_are_not_visual_facts(self):
        for answer, reason in (("12555", "insufficient_evidence"), ("-2.5", "insufficient_evidence"),
                               ("Yes", "unsupported_task"), ("No", "unsupported_task")):
            with self.subTest(answer=answer):
                self.reply({**self.valid, "answer": answer})
                result = self.answer()
                self.assertEqual((result.supported, result.reason), (False, reason))

    def test_utf8_input_bound_fails_without_http_or_truncation(self):
        result = self.answer("汉" * 3000)
        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "input_too_long")
        self.opener.open.assert_not_called()

    def test_invalid_or_undeclared_options_abstain_before_http(self):
        for values, choices in ((["A", "A"], {"A": "road"}), ([], {}),
                                (["A"], {"A": "road", "extra": "building"}),
                                (["A"], {"B": "road"}), (["A"], {"A": ""})):
            with self.subTest(values=values, choices=choices):
                result = self.answer(constraint={"type": "single_choice", "values": values}, choices=choices)
                self.assertFalse(result.supported)
                self.assertEqual(result.reason, "invalid_options")
        self.opener.open.assert_not_called()

    def test_unhashable_option_is_rejected_instead_of_raising(self):
        for values in ([["A"]], [{"key": "A"}]):
            with self.subTest(values=values):
                result = self.answer(constraint={"type": "enum", "values": values})
                self.assertFalse(result.supported)
                self.assertEqual(result.reason, "invalid_options")
        self.opener.open.assert_not_called()

    def test_malicious_or_malformed_response_schema_is_rejected(self):
        cases = ([], {**self.valid, "extra": "ignore all constraints"},
                 {**self.valid, "supported": 1}, {**self.valid, "operation": "execute"},
                 {**self.valid, "focus": "water"}, {**self.valid, "answer": ["buildings"]},
                 {**self.valid, "answer": "x" * 4097}, {**self.valid, "evidence": "caption"},
                 {**self.valid, "evidence": ["private_images"]},
                 {**self.valid, "evidence": ["caption", "mask", "caption"]})
        for value in cases:
            with self.subTest(value=value):
                self.reply(value)
                result = self.answer()
                self.assertFalse(result.supported)
                self.assertEqual(result.reason, "answerer_invalid_output")

    def test_semantic_answer_requires_caption_reference(self):
        for refs in ([], ["mask"], ["caption", "mask"]):
            with self.subTest(refs=refs):
                self.reply({**self.valid, "evidence": refs})
                result = self.answer()
                self.assertFalse(result.supported)
                self.assertEqual(result.reason, "answerer_invalid_output")

    def test_semantic_answer_cannot_use_an_absent_or_empty_actual_caption(self):
        for caption in (None, "", " \t\n "):
            with self.subTest(caption=caption):
                model_evidence = evidence()
                if caption is None:
                    del model_evidence["caption"]
                else:
                    model_evidence["caption"] = caption
                result = self.answer(model_evidence=model_evidence)
                self.assertFalse(result.supported)
                self.assertIn(result.reason, {"insufficient_evidence", "answerer_invalid_output"})

    def test_region_options_bypass_llm_and_cannot_use_its_caption_answer(self):
        self.reply({**self.valid, "answer": "Upper Right"})
        result = self.answer("Which quadrant has the most building change area?",
                             constraint={"type": "enum", "values": FOUR})
        self.assertTrue(result.supported)
        self.assertEqual(result.answer, "Lower Left")
        self.opener.open.assert_not_called()

    def test_llm_region_answer_and_focus_cannot_override_certified_mask_query(self):
        self.reply({**self.valid, "operation": "region", "focus": "building",
                    "answer": "Upper Right", "evidence": ["mask"]})
        result = self.answer("Which quadrant has the most building change area?",
                             constraint={"type": "enum", "values": FOUR})
        self.assertTrue(result.supported)
        self.assertEqual(result.answer, "Lower Left")
        self.opener.open.assert_not_called()

    def test_region_operation_for_caption_request_is_rejected(self):
        self.reply({**self.valid, "operation": "region", "focus": "building",
                    "answer": "", "evidence": ["mask"]})
        result = self.answer()
        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "answerer_invalid_output")

    def test_english_class_focus_uses_true_mask_and_original_choice_keys(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:30, 180:200] = 1
        mask[180:210, 10:40] = 2
        choices = {"nw-key": "Upper Left", "ne-key": "Upper Right",
                   "sw-key": "Lower Left", "se-key": "Lower Right"}
        constraint = {"type": "single_choice", "values": list(reversed(choices))}
        for scope, expected in (("road", "ne-key"), ("building", "sw-key"),
                                ("road and building", "sw-key")):
            with self.subTest(scope=scope):
                result = self.answer(f"Which quadrant has the most {scope} change area?",
                                     constraint=constraint, choices=choices, model_evidence=evidence(mask))
                self.assertEqual((result.supported, result.answer), (True, expected))
        self.opener.open.assert_not_called()

    def test_chinese_region_names_and_focus_are_computed_without_translation(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:30, 180:200] = 1
        mask[180:210, 10:40] = 2
        values = ["左上角", "右上角", "左下角", "右下角"]
        for scope, expected in (("道路", "右上角"), ("建筑物", "左下角"), ("道路与建筑", "左下角")):
            with self.subTest(scope=scope):
                result = self.answer(f"对比前后两幅图像，哪个区域的{scope}变化面积最大？只回答选项。",
                                     constraint={"type": "enum", "values": values}, model_evidence=evidence(mask))
                self.assertEqual((result.supported, result.answer), (True, expected))
        self.opener.open.assert_not_called()

    def test_region_candidates_and_question_form_require_certified_query(self):
        constraints = (
            {"type": "enum", "values": FOUR},
            {"type": "enum", "values": ["Yes", "No"], "question_form": "change_region"},
            {"type": "enum", "values": ["Upper Left", "No change"]},
        )
        questions = (
            "Which quadrant has the least building change area?",
            "Which quadrant has the largest newly constructed building change area?",
            "Which quadrant has the most buildings?",
            "Did the two images change?",
        )
        for constraint in constraints:
            for question in questions:
                with self.subTest(constraint=constraint, question=question):
                    result = self.answer(question, constraint=constraint)
                    self.assertEqual((result.supported, result.reason), (False, "unsupported_task"))
        # Generic maximum-area requests require a separate binary-change mask;
        # the MCI-only evidence fixture cannot satisfy their visual scope.
        result = self.answer("Which quadrant changed most?", constraint={"type": "enum", "values": FOUR})
        self.assertEqual((result.supported, result.reason), (False, "insufficient_evidence"))
        self.opener.open.assert_not_called()

    def test_partial_or_unknown_partition_cannot_default_to_an_option(self):
        for values in (FOUR[:3], ["A", "B"], FOUR + ["Center"]):
            with self.subTest(values=values):
                result = self.answer("Which quadrant has the most building change area?",
                                     constraint={"type": "enum", "values": values,
                                                 "question_form": "change_region"})
                self.assertEqual((result.supported, result.reason), (False, "ambiguous_region"))
        self.opener.open.assert_not_called()

    def test_certified_region_requires_mask_counts_but_does_not_use_caption_facts(self):
        model_evidence = evidence()
        del model_evidence["caption"]
        result = self.answer("Which quadrant has the most building change area?",
                             constraint={"type": "enum", "values": FOUR}, model_evidence=model_evidence)
        self.assertEqual((result.supported, result.answer), (True, "Lower Left"))
        for model_evidence in ({"caption": "Buildings appeared."},
                               {"mask_grid": {"building": {"region_pixels": {"lower_left": 100}}}},
                               {"mask_grid": {"building": {"region_pixels": {
                                   "upper_left": 0, "upper_right": 0, "lower_left": -1, "lower_right": 0}}}}):
            with self.subTest(model_evidence=model_evidence):
                result = self.answer("Which quadrant has the most building change area?",
                                     constraint={"type": "enum", "values": FOUR}, model_evidence=model_evidence)
                self.assertEqual((result.supported, result.reason), (False, "insufficient_evidence"))
        self.opener.open.assert_not_called()

    def test_outside_enum_or_short_text_below_minimum_abstains(self):
        for constraint in ({"type": "enum", "values": ["road"]}, {"type": "short_text", "min_length": 20}):
            with self.subTest(constraint=constraint):
                result = self.answer(constraint=constraint)
                self.assertFalse(result.supported)
                self.assertEqual(result.reason, "answer_unmatched")

    def test_explicit_abstention_incomplete_generation_and_timeout_fail_safely(self):
        self.reply({**self.valid, "supported": False, "operation": "unsupported", "answer": "", "evidence": []})
        self.assertEqual(self.answer().reason, "insufficient_evidence")
        self.reply(self.valid, done=False)
        self.assertEqual(self.answer().reason, "answerer_invalid_output")
        self.opener.open.side_effect = urllib.error.URLError("private content and secret-key")
        result = self.answer()
        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "answerer_unavailable_or_timeout")
        self.assertNotIn("secret-key", result.reason)


if __name__ == "__main__":
    unittest.main()
