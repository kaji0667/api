"""Mock model/CLIP boundaries; real synthetic geometry and phrase-map assignment."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_task_text import TaskInputError
from native_detection import NativeDetection, NativeTextEncoder
from ultralytics.data.augment import LetterBox
from ultralytics.data.base import BaseDataset


class NativeEncoderTests(unittest.TestCase):
    def setUp(self):
        self.core = Mock()
        self.core.tokenizer = SimpleNamespace(
            context_length=77, encode=Mock(side_effect=lambda text: text.split()),
        )
        raw = torch.arange(1, 769, dtype=torch.float32)
        self.original_feature = raw / raw.norm(p=2)
        self.core.encode.side_effect = lambda *, texts, batch_size, return_tokens: (
            self.original_feature.repeat(len(texts), 1)
        )
        factory_patch = patch("native_detection.OpenCLIPTextEncoder", return_value=self.core)
        self.factory = factory_patch.start()
        self.addCleanup(factory_patch.stop)
        self.encoder = NativeTextEncoder(device="cpu")

    def test_bare_descriptions_keep_case_sentences_and_have_one_row_each(self):
        descriptions = ["The SMALL warehouse.\nA river is above it.", "The bridge, except its road."]
        result = self.encoder.encode_descriptions(descriptions)
        prepared = ["The SMALL warehouse. A river is above it.", descriptions[1]]
        self.core.encode.assert_called_once_with(texts=prepared, batch_size=256, return_tokens=False)
        self.assertEqual(tuple(result.shape), (2, 768))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result.norm(dim=-1), torch.ones(2)))
        self.assertNotIn("satellite", self.core.encode.call_args.kwargs["texts"][0])
        self.factory.assert_called_once_with(
            model_name="ViT-L-14", pretrained="openai", device="cpu",
            precision="fp32", normalize=True,
        )

    def test_counting_uses_original_alias_cleanup_and_template(self):
        result = self.encoder.encode_counting("  WINDMILL  ")
        prompt = "a satellite remote sensing photo of wind turbine windmill"
        self.core.encode.assert_called_once_with(texts=[prompt], batch_size=256, return_tokens=False)
        self.assertEqual(tuple(result.shape), (1, 768))
        self.assertTrue(torch.allclose(result.norm(dim=-1), torch.ones(1)))
        self.core.tokenizer.encode.assert_called_once_with(prompt)

    def test_once_normalized_encoder_values_are_preserved_bitwise_in_cache(self):
        raw = torch.ones((1, 768), dtype=torch.float32)
        original = raw / raw.norm(p=2, dim=-1, keepdim=True)
        self.assertFalse(torch.equal(original, original / original.norm(p=2, dim=-1, keepdim=True)))
        self.core.encode.side_effect = None
        self.core.encode.return_value = original
        result = self.encoder.encode_descriptions(["the bridge"])
        self.assertTrue(torch.equal(result, original))
        self.assertTrue(torch.equal(self.encoder._cache["the bridge"], original[0]))
        result.zero_()
        cached = self.encoder.encode_descriptions(["the bridge"])
        self.assertTrue(torch.equal(cached, original))
        self.core.encode.assert_called_once_with(texts=["the bridge"], batch_size=256, return_tokens=False)

    def test_75_content_tokens_pass_and_76_fail_without_encoding_or_truncation(self):
        allowed = " ".join(["condition"] * 75)
        self.encoder.encode_descriptions([allowed])
        self.assertEqual(self.core.encode.call_args.kwargs["texts"], [allowed])
        self.core.encode.reset_mock()
        oversized = allowed + " final-condition"
        with self.assertRaises(TaskInputError) as context:
            self.encoder.encode_descriptions([oversized])
        self.assertEqual(context.exception.code, "text_context_exceeded")
        self.core.tokenizer.encode.assert_called_with(oversized)
        self.core.encode.assert_not_called()

    def test_counting_bound_includes_the_complete_template(self):
        query = " ".join(["object"] * 70)
        with self.assertRaises(TaskInputError) as context:
            self.encoder.encode_counting(query)
        self.assertEqual(context.exception.code, "text_context_exceeded")
        self.core.encode.assert_not_called()
        self.core.tokenizer.encode.assert_called_once_with(
            "a satellite remote sensing photo of " + query,
        )

    def test_oversized_member_rejects_entire_batch_before_encoding(self):
        with self.assertRaises(TaskInputError) as context:
            self.encoder.encode_descriptions(["short valid description", " ".join(["x"] * 76)])
        self.assertEqual(context.exception.code, "text_context_exceeded")
        self.core.encode.assert_not_called()

    def test_empty_descriptions_fail_without_model_encoding(self):
        for descriptions in ([], [""], [" \n\t "]):
            with self.subTest(descriptions=descriptions), self.assertRaises(TaskInputError):
                self.encoder.encode_descriptions(descriptions)
        self.core.encode.assert_not_called()

    def test_unverifiable_tokenizer_is_rejected(self):
        for tokenizer in (
            SimpleNamespace(context_length=77),
            SimpleNamespace(context_length=512, encode=lambda text: []),
        ):
            self.core.tokenizer = tokenizer
            with self.subTest(tokenizer=tokenizer), self.assertRaises(TaskInputError) as context:
                NativeTextEncoder(device="cpu")
            self.assertEqual(context.exception.code, "unsupported_text_tokenizer")
        self.core.encode.assert_not_called()

    def test_invalid_embeddings_fail_instead_of_entering_cache(self):
        self.core.encode.side_effect = None
        for features in (
            torch.ones((1, 12)), torch.zeros((1, 768)),
            torch.full((1, 768), float("nan")),
        ):
            self.core.encode.return_value = features
            with self.subTest(shape=features.shape), self.assertRaises(TaskInputError) as context:
                self.encoder.encode_descriptions(["the bridge"])
            self.assertEqual(context.exception.code, "invalid_text_embedding")
        self.assertEqual(self.encoder._cache, {})


class NativeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(return_value=torch.zeros((1, 5, 2)))
        self.model._last_text_outputs = {"phrase_logits": [
            torch.tensor([[[[1., 0.], [0., 5.]]]]),
        ]}
        self.backend = Mock(pt=True, nn_module=False, model=self.model, stride=32)
        backend_patch = patch("native_detection.AutoBackend", return_value=self.backend)
        self.backend_factory = backend_patch.start()
        self.addCleanup(backend_patch.stop)
        file_patch = patch("native_detection.Path.is_file", return_value=True)
        file_patch.start()
        self.addCleanup(file_patch.stop)
        self.encoder = Mock()
        self.encoder.encode_descriptions.return_value = torch.ones((1, 768)) / math.sqrt(768)
        self.detections = torch.tensor([
            [16., 16., 416., 416., .9, 0.],
            [416., 416., 816., 816., .2, 0.],
        ])
        nms_patch = patch("native_detection.non_max_suppression", return_value=[self.detections])
        self.nms = nms_patch.start()
        self.addCleanup(nms_patch.stop)
        text_patch = patch("native_detection.detection_text", return_value=
                           "The red warehouse at the left. A road lies above it.")
        self.task_text = text_patch.start()
        self.addCleanup(text_patch.stop)
        self.native = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder, spatial_weight=0)
        self.image = Image.new("RGB", (80, 80), (64, 128, 192))
        self.question = "Synthetic official source question with protocol packaging."

    def test_semantic_phrase_assignment_selects_lower_confidence_candidate(self):
        result = self.native.predict(self.image, self.question)
        self.assertEqual(result.grounding_index, 1)
        self.assertEqual(result.detection_count, 2)
        self.assertTrue(np.allclose(result.boxes, [[0, 0, 40, 40], [40, 40, 80, 80]]))
        self.assertTrue(np.allclose(result.scores, [.9, .2]))
        self.assertEqual(result.original_size, (80, 80))
        self.assertEqual(result.model_prompt, self.task_text.return_value)
        self.task_text.assert_called_once_with(self.question)
        self.encoder.encode_descriptions.assert_called_once_with([result.model_prompt])

    def test_absolute_corners_resolve_close_semantic_candidates_without_filtering(self):
        self.native = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder)
        self.nms.return_value = [torch.tensor([
            [16., 16., 416., 416., .9, 0.],
            [416., 16., 816., 416., .8, 0.],
            [16., 416., 416., 816., .2, 0.],
            [416., 416., 816., 816., .7, 0.],
        ])]
        cases = ("左上角的梯形草坪", "The lawn in the top-right corner of the image",
                 "左下角的梯形草坪，旁边有道路和房屋。", "The lawn in the bottom-right corner")
        for expected, description in enumerate(cases):
            with self.subTest(description=description):
                self.task_text.return_value = description
                self.encoder.encode_descriptions.reset_mock()
                with patch("native_detection.score_matrix_from_phrase_maps", return_value=torch.ones((1, 4))):
                    result = self.native.predict(self.image, self.question)
                self.assertEqual(result.grounding_index, expected)
                self.assertEqual(result.detection_count, 4)
                np.testing.assert_allclose(result.scores, [.9, .8, .2, .7])
                np.testing.assert_array_equal(result.boxes, [[0, 0, 40, 40], [40, 0, 80, 40],
                                                           [0, 40, 40, 80], [40, 40, 80, 80]])
                self.assertEqual(result.model_prompt, description)
                self.encoder.encode_descriptions.assert_called_once_with([description])
        self.assertEqual(self.nms.call_args.kwargs["conf_thres"], .25)
        self.assertEqual(self.nms.call_args.kwargs["iou_thres"], .7)

    def test_soft_bonus_can_resolve_near_tie_but_not_override_strong_semantics(self):
        self.native = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder)
        self.task_text.return_value = "图像右下角的草坪"
        for scores, expected in (([[1.02, 1.]], 1), ([[1.3, 1.]], 0)):
            with self.subTest(scores=scores):
                with patch("native_detection.score_matrix_from_phrase_maps", return_value=torch.tensor(scores)):
                    result = self.native.predict(self.image, self.question)
                self.assertEqual(result.grounding_index, expected)

    def test_zero_weight_and_relative_or_ambiguous_text_keep_semantic_selection(self):
        self.native = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder)
        cases = ("The lawn to the lower left of the warehouse", "右上角仓库旁的汽车",
                 "草坪不在左下角", "The trapezoidal lawn")
        for description in cases:
            with self.subTest(description=description):
                self.task_text.return_value = description
                with patch("native_detection.score_matrix_from_phrase_maps", return_value=torch.tensor([[1.02, 1.]])):
                    result = self.native.predict(self.image, self.question)
                self.assertEqual(result.grounding_index, 0)
        self.native.spatial_weight = 0
        self.task_text.return_value = "图像右下角的草坪"
        with patch("native_detection.parse_absolute_position") as parse, \
                patch("native_detection.score_matrix_from_phrase_maps", return_value=torch.tensor([[1.02, 1.]])):
            result = self.native.predict(self.image, self.question)
        parse.assert_not_called()
        self.assertEqual(result.grounding_index, 0)

    def test_spatial_bonus_uses_restored_original_coordinates_on_panorama(self):
        self.native = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder)
        self.task_text.return_value = "The building in the bottom-right corner of the image"
        image = Image.new("RGB", (10000, 101))
        original_boxes = torch.tensor([[1000., 10., 3000., 30.], [7000., 70., 9000., 90.]])
        gains = torch.tensor([.16, 17 / 101, .16, 17 / 101])
        network = original_boxes * gains + 16
        self.nms.return_value = [torch.cat((network, torch.tensor([[.9, 0.], [.2, 0.]])), dim=1)]
        with patch.object(self.native, "_preprocess", return_value=(
                torch.zeros((1, 3, 64, 1632)), ((17 / 101, .16), (16, 16)))), \
                patch("native_detection.score_matrix_from_phrase_maps", return_value=torch.tensor([[1.02, 1.]])), \
                patch.object(self.native, "_position_adjusted_scores", wraps=self.native._position_adjusted_scores) as adjust:
            result = self.native.predict(image, self.question)
        torch.testing.assert_close(adjust.call_args.args[1], original_boxes, atol=.002, rtol=0)
        self.assertEqual(adjust.call_args.args[2], (101, 10000))
        self.assertEqual(result.grounding_index, 1)

    def test_presence_skips_spatial_parser_even_with_enabled_weight(self):
        self.native = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder)
        self.task_text.return_value = "图像右下角的草坪"
        with patch("native_detection.parse_absolute_position") as parse:
            result = self.native.predict(self.image, self.question, require_grounding=False)
        parse.assert_not_called()
        self.assertIsNone(result.grounding_index)
        self.assertEqual(result.detection_count, 2)

    def test_invalid_spatial_weight_fails_before_backend_loading(self):
        self.backend_factory.reset_mock()
        for weight in (-.1, 1.01, float("nan"), float("inf"), True, "0.15"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder, spatial_weight=weight)
        self.backend_factory.assert_not_called()

    def test_candidate_threshold_rejects_invalid_values_before_backend_loading(self):
        self.backend_factory.reset_mock()
        for value in (0, 1, -.1, 1.01, float("nan"), float("inf"), True, "0.25", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder, candidate_confidence=value)
        self.backend_factory.assert_not_called()
        with self.assertRaises(ValueError):
            self.native.candidate_confidence = False
        self.assertEqual(self.native.candidate_confidence, .25)

    def test_explicit_default_threshold_has_identical_predictions_and_nms(self):
        default = self.native.predict(self.image, self.question, return_phrase_scores=True)
        explicit = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder,
                                   spatial_weight=0, candidate_confidence=.25)
        other = explicit.predict(self.image, self.question, return_phrase_scores=True)
        self.assertEqual(default.grounding_index, other.grounding_index)
        np.testing.assert_array_equal(default.boxes, other.boxes)
        np.testing.assert_array_equal(default.phrase_scores, other.phrase_scores)
        np.testing.assert_array_equal(default.scores, other.scores)
        self.assertEqual(self.nms.call_args.kwargs, {
            "conf_thres": .25, "iou_thres": .7, "multi_label": False, "agnostic": True, "max_det": 300,
        })

    def test_real_nms_three_thresholds_use_fresh_forwards_and_preserve_shared_geometry(self):
        from ultralytics.utils.ops import non_max_suppression as real_nms
        raw = torch.tensor([[[150., 350., 650.], [150., 350., 650.],
                             [100., 100., 100.], [100., 100., 100.], [.30, .15, .07]]])
        self.model.side_effect = lambda *args, **kwargs: raw.clone()
        self.nms.side_effect = real_nms
        results = []
        original_model, original_encoder = self.native.model, self.native.encoder
        with patch.dict("os.environ", {"HFSA_API_DIAGNOSTICS": "1"}):
            for threshold, expected_count in ((.25, 1), (.10, 2), (.05, 3)):
                self.native.candidate_confidence = threshold
                result = self.native.predict(self.image, self.question, return_phrase_scores=True)
                results.append(result)
                self.assertEqual(result.detection_count, expected_count)
                self.assertEqual(result.diagnostics["pre_nms_above_threshold"], expected_count)
                self.assertEqual(result.diagnostics["confidence_threshold"], threshold)
                self.assertEqual(result.diagnostics["nms_candidate_count"], expected_count)
                self.assertEqual(self.nms.call_args.kwargs["conf_thres"], threshold)
                self.assertIs(self.native.model, original_model)
                self.assertIs(self.native.encoder, original_encoder)
        self.assertEqual(self.model.call_count, 3)
        self.assertEqual(float(raw[0, 0, 0]), 150.)  # NMS in-place conversion never contaminated the source.
        np.testing.assert_allclose(results[2].boxes, [[8.4, 8.4, 18.4, 18.4],
                                                   [28.4, 28.4, 38.4, 38.4], [58.4, 58.4, 68.4, 68.4]], atol=1e-5)
        for result in results[:2]:
            np.testing.assert_array_equal(result.boxes, results[2].boxes[:len(result.boxes)])
            np.testing.assert_array_equal(result.phrase_scores, results[2].phrase_scores[:len(result.boxes)])

    def test_custom_threshold_diagnostic_matches_empty_candidate_pool(self):
        self.native.candidate_confidence = .10
        self.model.return_value = torch.tensor([[[1., 2.], [1., 2.], [1., 1.], [1., 1.], [.07, .09]]])
        self.nms.return_value = [torch.empty((0, 6))]
        with patch.dict("os.environ", {"HFSA_API_DIAGNOSTICS": "1"}):
            result = self.native.predict(self.image, self.question, return_phrase_scores=True)
        self.assertEqual(result.diagnostics["confidence_threshold"], .10)
        self.assertEqual(result.diagnostics["pre_nms_above_threshold"], 0)
        self.assertEqual(self.nms.call_args.kwargs["conf_thres"], .10)
        self.assertEqual(result.detection_count, 0)

    def test_opt_in_diagnostic_observes_below_threshold_candidates_without_changing_nms(self):
        raw = torch.zeros((1, 5, 2))
        raw[0, 4] = torch.tensor([.10, .24])
        self.model.return_value = raw
        self.nms.return_value = [torch.empty((0, 6))]
        with patch.dict("os.environ", {"HFSA_API_DIAGNOSTICS": "1"}):
            result = self.native.predict(self.image, self.question)
        self.assertEqual(result.detection_count, 0)
        self.assertIsNone(result.grounding_index)
        self.assertEqual(result.diagnostics["input_hw"], [832, 832])
        self.assertEqual(result.diagnostics["original_hw"], [80, 80])
        self.assertAlmostEqual(result.diagnostics["max_confidence"], .24, places=6)
        self.assertEqual(result.diagnostics["raw_anchor_count"], 2)
        self.assertEqual(result.diagnostics["pre_nms_above_threshold"], 0)
        self.assertEqual(result.diagnostics["nms_candidate_count"], 0)
        self.assertEqual(self.nms.call_args.kwargs["conf_thres"], .25)

    def test_diagnostic_is_disabled_by_default_and_does_not_change_selection(self):
        with patch.dict("os.environ", {"HFSA_API_DIAGNOSTICS": "0"}):
            plain = self.native.predict(self.image, self.question)
        with patch.dict("os.environ", {"HFSA_API_DIAGNOSTICS": "1"}):
            diagnostic = self.native.predict(self.image, self.question)
        self.assertIsNone(plain.diagnostics)
        self.assertEqual(plain.grounding_index, diagnostic.grounding_index)
        np.testing.assert_array_equal(plain.boxes, diagnostic.boxes)
        np.testing.assert_array_equal(plain.scores, diagnostic.scores)

    def test_forward_matches_single_object_cache_conditions_without_gt(self):
        self.native.predict(self.image, self.question)
        tensor = self.model.call_args.args[0]
        conditions = self.model.call_args.kwargs
        self.assertEqual(tuple(tensor.shape), (1, 3, 832, 832))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(float(tensor[0, 0, 400, 400]), 64 / 255, places=6)
        self.assertAlmostEqual(float(tensor[0, 2, 400, 400]), 192 / 255, places=6)
        self.assertIs(conditions["augment"], False)
        self.assertEqual(tuple(conditions["txt_vec"].shape), (1, 1, 768))
        self.assertTrue(torch.equal(conditions["txt_token_mask"], torch.ones((1, 1), dtype=torch.bool)))
        self.assertTrue(torch.equal(conditions["txt_phrase_weight"], torch.tensor([[2.]])))
        self.assertTrue(torch.equal(conditions["txt_dependency_adj"], torch.ones((1, 1, 1))))
        self.assertEqual(set(conditions), {
            "augment", "txt_vec", "txt_token_mask", "txt_phrase_weight", "txt_dependency_adj",
        })
        self.assertEqual(self.nms.call_args.kwargs, {
            "conf_thres": .25, "iou_thres": .7, "multi_label": False,
            "agnostic": True, "max_det": 300,
        })
        self.assertIs(self.backend_factory.call_args.kwargs["fp16"], False)
        self.model.eval.assert_called_once_with()

    def test_letterbox_padding_is_removed_in_native_coordinates(self):
        self.nms.return_value = [torch.tensor([[116., 108., 716., 308., .8, 0.]])]
        self.model._last_text_outputs = {"phrase_logits": [torch.ones((1, 1, 2, 2))]}
        result = self.native.predict(Image.new("RGB", (80, 40)), self.question)
        self.assertEqual(result.original_size, (40, 80))
        self.assertEqual(tuple(self.model.call_args.args[0].shape), (1, 3, 416, 832))
        self.assertTrue(np.allclose(result.boxes, [[10., 10., 70., 30.]]))
        self.assertEqual(result.grounding_index, 0)

    def test_noninteger_resize_prediction_restores_each_axis_once(self):
        # Independently specified BaseDataset resize: 953x1047 -> 729x800;
        # rect input is 768x832, with left=16/top=19. Do not use scale_boxes
        # to construct the expected native-coordinate answer.
        original_box = torch.tensor([[101., 203., 941., 761.]])
        network_box = original_box * torch.tensor([[800 / 1047, 729 / 953, 800 / 1047, 729 / 953]])
        network_box += torch.tensor([[16., 19., 16., 19.]])
        self.nms.return_value = [torch.cat((network_box, torch.tensor([[.8, 0.]])), dim=1)]
        self.model._last_text_outputs = {"phrase_logits": [torch.ones((1, 1, 2, 2))]}
        result = self.native.predict(Image.new("RGB", (1047, 953)), self.question)
        self.assertEqual(tuple(self.model.call_args.args[0].shape), (1, 3, 768, 832))
        np.testing.assert_allclose(result.boxes, original_box.numpy(), atol=1e-4, rtol=0)
        self.assertEqual(result.grounding_index, 0)

    def test_preprocess_matches_original_dataset_rect_and_letterbox(self):
        for height, width, stride, image_size in (
            (80, 80, 32, 800), (40, 80, 32, 800), (57, 123, 32, 800), (127, 59, 32, 800),
            (800, 701, 32, 800), (953, 1047, 64, 800), (953, 1047, 32, 1600),
            (491, 736, 32, 1600), (953, 1047, 32, 1056),
        ):
            with self.subTest(height=height, width=width, stride=stride, image_size=image_size):
                pixels = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
                image = (pixels % 256).astype(np.uint8)
                reference = BaseDataset.__new__(BaseDataset)
                reference.ims = [None]
                reference.im_files = ["synthetic.png"]
                reference.npy_files = [Path("synthetic.npy")]
                reference.augment = False
                reference.imgsz = image_size
                reference.ni = 1
                reference.batch_size = 1
                reference.labels = [{"shape": (height, width)}]
                reference.stride = stride
                reference.pad = .5
                with patch("ultralytics.data.base.cv2.imread", return_value=image), \
                        patch.object(Path, "exists", return_value=False):
                    resized, original_hw, resized_hw = reference.load_image(0)
                reference.set_rectangle()
                reference_image = LetterBox(new_shape=reference.batch_shapes[0], scaleup=False,
                                            stride=stride)(image=resized)
                reference_tensor = torch.from_numpy(
                    np.ascontiguousarray(reference_image[:, :, ::-1].transpose(2, 0, 1))
                ).float().unsqueeze(0) / 255
                rect_h, rect_w = reference.batch_shapes[0]
                reference_ratio_pad = (
                    (resized_hw[0] / original_hw[0], resized_hw[1] / original_hw[1]),
                    (int(round((rect_w - resized_hw[1]) / 2 - .1)),
                     int(round((rect_h - resized_hw[0]) / 2 - .1))),
                )
                actual_tensor, actual_ratio_pad = self.native._preprocess(image, stride=stride, image_size=image_size)
                self.assertTrue(torch.equal(actual_tensor, reference_tensor))
                self.assertEqual(actual_ratio_pad, reference_ratio_pad)

    def test_adaptive_size_preserves_small_images_and_caps_large_assets(self):
        high_resolution = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder, max_image_size=1600)
        for original_hw, expected in (((512, 512), 800), ((800, 800), 800),
                                      ((801, 700), 832), ((953, 1047), 1056),
                                      ((4096, 4097), 1600), ((4912, 7360), 1600)):
            with self.subTest(original_hw=original_hw):
                self.assertEqual(high_resolution.image_size_for(original_hw), expected)
                self.assertEqual(self.native.image_size_for(original_hw), 800)

    def test_large_asset_matching_returns_original_coordinates_once(self):
        high_resolution = NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder, max_image_size=1600)
        self.nms.return_value = [torch.tensor([[116., 116., 316., 316., .8, 0.]])]
        self.model._last_text_outputs = {"phrase_logits": [torch.ones((1, 1, 2, 2))]}
        result = high_resolution.predict(Image.new("RGB", (3200, 1600)), self.question)
        self.assertEqual(tuple(self.model.call_args.args[0].shape), (1, 3, 832, 1632))
        self.assertEqual(result.original_size, (1600, 3200))
        self.assertEqual(result.grounding_index, 0)
        self.assertTrue(np.allclose(result.boxes, [[200., 200., 600., 600.]]))
        self.encoder.encode_descriptions.assert_called_once_with([self.task_text.return_value])

    def test_invalid_size_cap_fails_before_loading_models(self):
        self.backend_factory.reset_mock()
        for size in (0, 640, 801, 1600.0, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                NativeDetection("synthetic.pt", device="cpu", encoder=self.encoder, max_image_size=size)
        self.backend_factory.assert_not_called()

    def test_prediction_uses_backend_stride_for_rect_shape(self):
        self.backend.stride = 64
        self.native.predict(self.image, self.question, require_grounding=False)
        self.assertEqual(tuple(self.model.call_args.args[0].shape), (1, 3, 832, 832))
        self.backend.stride = 128
        self.native.predict(self.image, self.question, require_grounding=False)
        self.assertEqual(tuple(self.model.call_args.args[0].shape), (1, 3, 896, 896))

    def test_paths_use_original_opencv_decoder_and_bad_files_fail(self):
        pixels = np.full((9, 13, 3), [10, 20, 30], dtype=np.uint8)
        with patch("native_detection.cv2.imread", return_value=pixels) as imread, \
                patch("native_detection.Image.open") as pil_open:
            for image in ("synthetic.jpg", Path("synthetic.jpg")):
                self.assertTrue(np.array_equal(self.native._image(image), pixels))
            self.assertEqual(imread.call_count, 2)
            imread.assert_called_with("synthetic.jpg")
            pil_open.assert_not_called()
        with patch("native_detection.cv2.imread", return_value=None):
            with self.assertRaises(TaskInputError) as context:
                self.native._image("synthetic-missing.jpg")
        self.assertEqual(context.exception.code, "invalid_detection_image")

    def test_empty_detection_returns_empty_result_without_guessing(self):
        self.nms.return_value = [torch.empty((0, 6))]
        self.model._last_text_outputs = {}
        result = self.native.predict(self.image, self.question)
        self.assertEqual(result.detection_count, 0)
        self.assertEqual(result.boxes.shape, (0, 4))
        self.assertEqual(result.scores.shape, (0,))
        self.assertIsNone(result.grounding_index)

    def test_grounding_without_phrase_maps_fails_without_confidence_fallback(self):
        self.model._last_text_outputs = {}
        with self.assertRaises(TaskInputError) as context:
            self.native.predict(self.image, self.question)
        self.assertEqual(context.exception.code, "detection_phrase_maps_missing")

    def test_presence_does_not_require_phrase_maps_or_assignment(self):
        self.model._last_text_outputs = {}
        with patch("native_detection.score_matrix_from_phrase_maps") as score:
            result = self.native.predict(self.image, self.question, require_grounding=False)
        score.assert_not_called()
        self.assertEqual(result.detection_count, 2)
        self.assertIsNone(result.grounding_index)

    def test_invalid_maps_or_assignment_are_rejected(self):
        for levels in ([torch.zeros((1, 2, 2, 2))], [torch.full((1, 1, 2, 2), float("nan"))]):
            self.model._last_text_outputs = {"phrase_logits": levels}
            with self.subTest(levels=len(levels)), self.assertRaises(TaskInputError) as context:
                self.native.predict(self.image, self.question)
            self.assertEqual(context.exception.code, "invalid_detection_phrase_maps")
        self.model._last_text_outputs = {"phrase_logits": [torch.ones((1, 1, 2, 2))]}
        with patch("native_detection.assign_by_score_matrix", return_value=(torch.tensor([0]), torch.tensor([2]))):
            with self.assertRaises(TaskInputError) as context:
                self.native.predict(self.image, self.question)
        self.assertEqual(context.exception.code, "invalid_detection_matching")

    def test_oversize_text_error_prevents_model_forward(self):
        self.encoder.encode_descriptions.side_effect = TaskInputError("text_context_exceeded", "Fixed safe error")
        with self.assertRaises(TaskInputError) as context:
            self.native.predict(self.image, self.question)
        self.assertEqual(context.exception.code, "text_context_exceeded")
        self.model.assert_not_called()
        self.nms.assert_not_called()

    def test_invalid_candidates_and_embedding_fail_explicitly(self):
        self.nms.return_value = [torch.full((1, 6), float("inf"))]
        with self.assertRaises(TaskInputError) as context:
            self.native.predict(self.image, self.question)
        self.assertEqual(context.exception.code, "invalid_detection_result")
        self.encoder.encode_descriptions.return_value = torch.ones((1, 16))
        self.model.reset_mock()
        with self.assertRaises(TaskInputError) as context:
            self.native.predict(self.image, self.question)
        self.assertEqual(context.exception.code, "invalid_text_embedding")
        self.model.assert_not_called()


class NativeDetectionGeometryTests(unittest.TestCase):
    def test_independent_original_boxes_roundtrip_with_axis_specific_resize(self):
        # Known ceil-resize dimensions, including portrait and panoramic images.
        for height, width, resized_h, resized_w in (
            (953, 1047, 729, 800), (491, 736, 1068, 1600),
            (101, 10000, 17, 1600), (10000, 101, 1600, 17),
            (3200, 3200, 1600, 1600),
        ):
            with self.subTest(height=height, width=width):
                original = torch.tensor([
                    [0., 0., float(width), float(height)],
                    [.1 * width, .2 * height, .9 * width, .8 * height],
                ], dtype=torch.float64)
                gain_h, gain_w = resized_h / height, resized_w / width
                # Forward mapping is specified independently of the recovery.
                network = original * torch.tensor(
                    [gain_w, gain_h, gain_w, gain_h], dtype=torch.float64,
                ) + torch.tensor([16., 7., 16., 7.], dtype=torch.float64)
                saved = network.clone()
                actual = NativeDetection._restore_boxes(network, (height, width), ((gain_h, gain_w), (16, 7)))
                torch.testing.assert_close(actual, original, atol=1e-10, rtol=0)
                torch.testing.assert_close(network, saved, atol=0, rtol=0)

    def test_uniform_height_gain_cannot_recover_panorama_x_coordinates(self):
        original = torch.tensor([[1000., 20.2, 9000., 80.8]], dtype=torch.float64)
        gains = torch.tensor([.16, 17 / 101, .16, 17 / 101], dtype=torch.float64)
        network = original * gains + 16
        legacy = (network - 16) / (17 / 101)
        self.assertGreater(float((legacy[:, [0, 2]] - original[:, [0, 2]]).abs().max()), 400)
        actual = NativeDetection._restore_boxes(network, (101, 10000), ((17 / 101, .16), (16, 16)))
        torch.testing.assert_close(actual, original, atol=1e-10, rtol=0)

    def test_restoration_clips_original_bounds_and_accepts_empty_boxes(self):
        network = torch.tensor([[-4., 6., 244., 116.]])
        actual = NativeDetection._restore_boxes(network, (50, 80), ((2, 3), (4, 6)))
        torch.testing.assert_close(actual, torch.tensor([[0., 0., 80., 50.]]), atol=0, rtol=0)
        empty = NativeDetection._restore_boxes(torch.empty((0, 4)), (50, 80), ((2, 3), (4, 6)))
        self.assertEqual(tuple(empty.shape), (0, 4))

    def test_invalid_geometry_is_rejected_before_division(self):
        boxes = torch.ones((1, 4))
        for original_hw, ratio_pad, candidates in (
            ((0, 80), ((2, 3), (4, 6)), boxes),
            ((50, True), ((2, 3), (4, 6)), boxes),
            ((50, 80), ((0, 3), (4, 6)), boxes),
            ((50, 80), ((2, float("nan")), (4, 6)), boxes),
            ((50, 80), ((2, 3), (-4, 6)), boxes),
            ((50, 80), ((2,), (4, 6)), boxes),
            ((50, 80), ((2, 3), (4, 6)), torch.ones((1, 3))),
            ((50, 80), ((2, 3), (4, 6)), torch.ones((1, 4), dtype=torch.long)),
            ((50, 80), ((2, 3), (4, 6)), torch.full((1, 4), float("inf"))),
        ):
            with self.subTest(original_hw=original_hw, ratio_pad=ratio_pad), self.assertRaises(TaskInputError) as context:
                NativeDetection._restore_boxes(candidates, original_hw, ratio_pad)
            self.assertEqual(context.exception.code, "invalid_detection_geometry")


if __name__ == "__main__":
    unittest.main()
