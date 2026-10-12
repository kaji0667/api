"""One-time native/reference parity checks using the same loaded API models.

No labels, annotations, result identifiers or gold answers are read. All queries
are synthetic. Logs contain only comparison statistics, never image pixels,
question/description strings, class predictions, boxes, answers or credentials.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _close(label: str, actual: Any, expected: Any, *, atol: float = 1e-6, rtol: float = 1e-5) -> float:
    actual_tensor = torch.as_tensor(actual).detach().cpu()
    expected_tensor = torch.as_tensor(expected).detach().cpu()
    if actual_tensor.shape != expected_tensor.shape:
        raise AssertionError(f"Native parity shape mismatch: {label}")
    try:
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol, check_dtype=False)
    except AssertionError:
        # Do not expose tensor values or model answers in an assertion traceback.
        raise AssertionError(f"Native parity numeric mismatch: {label}") from None
    if not actual_tensor.numel():
        return 0.0
    return float((actual_tensor.double() - expected_tensor.double()).abs().max())


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(f"Native parity contract mismatch: {label}")


def _first_tensor(output: Any) -> torch.Tensor:
    while isinstance(output, (list, tuple)) and output:
        output = output[0]
    _require(isinstance(output, torch.Tensor), "forward_tensor")
    return output


@contextmanager
def _capture_forward(model: Any):
    """Observe the native call without replacing its input preparation or model."""
    captured: dict[str, Any] = {}

    def before(_module: Any, args: Any, kwargs: Any) -> None:
        captured["image"] = args[0].detach().cpu().clone()
        captured["conditions"] = {
            key: value.detach().cpu().clone()
            for key, value in kwargs.items() if isinstance(value, torch.Tensor)
        }

    def after(module: Any, _args: Any, output: Any) -> None:
        captured["raw"] = _first_tensor(output).detach().cpu().clone()
        last = getattr(module, "_last_text_outputs", None)
        maps = last.get("phrase_logits") if isinstance(last, dict) else None
        captured["phrase_maps"] = [value.detach().cpu().clone() for value in maps] if isinstance(maps, list) else []

    before_handle = model.register_forward_pre_hook(before, with_kwargs=True)
    after_handle = model.register_forward_hook(after)
    try:
        yield captured
    finally:
        before_handle.remove()
        after_handle.remove()


def _request(question: str, kind: str, height: int, width: int, **extra: Any) -> dict[str, Any]:
    return {
        "question": question, "image_height": height, "image_width": width,
        "response_constraint": {"type": kind, **extra},
    }


@contextmanager
def _backend(adapter: Any, backend: str, refseg: Any = None):
    previous_backend, previous_refseg = adapter.grounding_backend, adapter.refseg
    adapter.grounding_backend = backend
    if refseg is not None:
        adapter.refseg = refseg
    try:
        yield
    finally:
        adapter.grounding_backend, adapter.refseg = previous_backend, previous_refseg


def _api_box_check(adapter: Any, image: Path, question: str, expected: list[int] | None,
                   empty_code: str, height: int, width: int) -> None:
    from hfsa_adapter import AdapterFailure, QuestionPlan

    request = _request(question, "bbox", height, width)
    try:
        actual = adapter._predict_plan(request, image, QuestionPlan("visual_grounding", question))
    except AdapterFailure as error:
        _require(expected is None and error.code == empty_code, "bbox_failure_mapping")
    else:
        _require(expected is not None and actual == expected, "bbox_answer_mapping")


def _classification(adapter: Any, image: Path, rgb: Image.Image) -> dict[str, Any]:
    from hfsa_adapter import QuestionPlan
    from tasks.classification.data import SceneDataModule

    native = adapter.classification
    _require(not native.network.training, "classification_eval")
    tensor = SceneDataModule.build_transform(640, train=False)(rgb).unsqueeze(0).to(native.device)
    with torch.inference_mode():
        reference_logits = native.network(tensor)
        reference_probabilities = reference_logits.softmax(dim=1)[0]
    with _capture_forward(native.network) as observed:
        result = native.predict(image, topk=len(native.class_names))
    input_error = _close("classification_input", observed["image"], tensor)
    logits_error = _close("classification_logits", observed["raw"], reference_logits)
    restored = torch.zeros(len(native.class_names), dtype=torch.float32)
    seen: set[int] = set()
    for row in result["predictions"]:
        class_id = int(row["class_id"])
        _require(0 <= class_id < len(native.class_names) and class_id not in seen, "classification_class_order")
        _require(row["class_name"] == native.class_names[class_id], "classification_class_name")
        seen.add(class_id)
        restored[class_id] = float(row["probability"])
    _require(len(seen) == len(native.class_names), "classification_complete_probabilities")
    probability_error = _close("classification_probabilities", restored, reference_probabilities)
    height, width = rgb.height, rgb.width
    question = "Classify the remote sensing scene."
    direct_top_index = int(reference_probabilities.topk(1).indices[0])
    mapped = adapter._predict_plan(
        _request(question, "short_text", height, width), image, QuestionPlan("classification", question)
    )
    _require(mapped == native.class_names[direct_top_index], "classification_short_text_mapping")
    choices = {f"C{index}": name for index, name in enumerate(native.class_names)}
    request = _request(question, "single_choice", height, width, values=list(choices))
    request["choices"] = choices
    mapped = adapter._predict_plan(request, image, QuestionPlan("classification", question))
    _require(mapped == f"C{int(reference_probabilities.argmax())}", "classification_option_probability_mapping")
    return {
        "checked": True, "class_count": len(native.class_names), "input_hw": [640, 640],
        "input_max_abs_error": input_error, "logits_max_abs_error": logits_error,
        "probability_max_abs_error": probability_error, "api_mapping_checked": True,
    }


def _counting(adapter: Any, image: Path, rgb: Image.Image) -> dict[str, Any]:
    from hfsa_adapter import QuestionPlan
    from tasks.counting.inference import CountingImagePreprocessor, ObjectCounter
    from tasks.detection.inference import TextPromptEncoder
    from ultralytics.utils.ops import scale_boxes

    native = adapter.counting
    _require(not native.counter.model.training, "counting_eval")
    target = "windmill"
    # Execute the original prompt/tokenization/normalization implementation while
    # reusing the already-loaded FP32 CLIP model; __init__ would load another.
    original_encoder = TextPromptEncoder.__new__(TextPromptEncoder)
    original_encoder.device = native.device
    original_encoder.prompt_template = "a satellite remote sensing photo of {class_name}"
    original_encoder.clip_model = native.encoder.encoder.model
    original_encoder.tokenizer = native.encoder.encoder.tokenizer
    original_encoder.cache = {}
    reference_features = original_encoder.encode(target).squeeze(1)
    native_features = native.encoder.encode_counting(target)
    feature_error = _close("counting_text_features", native_features, reference_features)

    bgr = cv2.imread(str(image))
    _require(bgr is not None, "counting_original_decode")
    tensor, _, _ = CountingImagePreprocessor(800)(bgr)
    reference_counter = ObjectCounter(native.counter.model, original_encoder, native.device)
    with _capture_forward(native.counter.model) as reference_observed:
        reference = reference_counter.count(tensor, target, conf_thres=.15, iou_thres=.5, max_det=300)
    reference_boxes = reference["pred_boxes"].detach().cpu().clone()
    if len(reference_boxes):
        scale_boxes(tensor.shape[2:], reference_boxes, bgr.shape[:2])
    question = "How many windmills are there in the image?"
    with _capture_forward(native.counter.model) as observed:
        result = native.predict(image, question)
    input_error = _close("counting_input", observed["image"], tensor)
    raw_error = _close("counting_raw_predictions", observed["raw"], reference_observed["raw"], atol=2e-3)
    _close("counting_forward_text_features", observed["conditions"]["txt_vec"], reference_features.unsqueeze(1))
    boxes_error = _close("counting_boxes", result.boxes, reference_boxes, atol=2e-3)
    scores_error = _close("counting_scores", result.scores, reference["pred_scores"], atol=1e-6)
    _require(result.count == int(reference["pred_count"]), "counting_nms_count")
    mapped = adapter._predict_plan(
        _request(question, "integer", rgb.height, rgb.width, minimum=0), image, QuestionPlan("counting", question)
    )
    _require(mapped == str(int(reference["pred_count"])), "counting_integer_mapping")
    return {
        "checked": True, "candidate_count": len(reference_boxes), "input_hw": [800, 800],
        "input_max_abs_error": input_error, "text_feature_max_abs_error": feature_error,
        "raw_max_abs_error": raw_error,
        "boxes_max_abs_error": boxes_error, "scores_max_abs_error": scores_error,
        "api_mapping_checked": True,
    }


def _original_detection_image(image: Path, stride: int, image_size: int = 800) -> tuple[torch.Tensor, tuple[Any, Any], tuple[int, int]]:
    """Original rect=True, pad=.5 evaluator preparation, without any labels/GT."""
    from ultralytics.data.augment import LetterBox
    from ultralytics.data.base import BaseDataset

    decoded = cv2.imread(str(image))
    _require(decoded is not None, "detection_reference_decode")
    original_hw = tuple(int(value) for value in decoded.shape[:2])
    no_cache = image.with_name("__hfsa_native_parity_no_cache__.npy")
    _require(not no_cache.exists(), "detection_reference_no_cache")
    dataset = BaseDataset.__new__(BaseDataset)
    dataset.imgsz, dataset.augment, dataset.prefix = image_size, False, ""
    dataset.ims, dataset.npy_files, dataset.im_files = [None], [no_cache], [str(image)]
    dataset.labels = [{"shape": original_hw}]
    dataset.batch_size, dataset.ni, dataset.stride, dataset.pad = 1, 1, stride, .5
    BaseDataset.set_rectangle(dataset)
    resized, loaded_hw, resized_hw = BaseDataset.load_image(dataset, 0)
    shape = tuple(int(value) for value in dataset.batch_shapes[0])
    letterboxed = LetterBox(new_shape=shape, auto=False, scaleup=False, stride=stride)(image=resized)
    # This is the same ratio_pad metadata emitted by get_image_and_label and
    # LetterBox; the shape-only stub never invokes label loading or formatting.
    ratio_pad = (
        (resized_hw[0] / loaded_hw[0], resized_hw[1] / loaded_hw[1]),
        (int(round((shape[1] - resized_hw[1]) / 2 - .1)), int(round((shape[0] - resized_hw[0]) / 2 - .1))),
    )
    tensor = torch.from_numpy(np.ascontiguousarray(letterboxed[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255
    return tensor, ratio_pad, original_hw


def _detection_reference(native: Any, tensor: torch.Tensor, description: str,
                         ratio_pad: tuple[Any, Any], original_hw: tuple[int, int],
                         match: bool) -> dict[str, Any]:
    from tasks.detection.test import _run_text_inference
    from text_encoder.matching import assign_by_score_matrix, score_matrix_from_phrase_maps
    from text_encoder.phrase_classifier import classify_phrase, get_phrase_type_weight
    from ultralytics.utils.ops import non_max_suppression, scale_boxes

    features = native.encoder.encoder.encode([description], batch_size=1, return_tokens=False)
    features = features.to(device=native.device, dtype=tensor.dtype).unsqueeze(0)
    conditions = {
        "txt_vec": features,
        "text_token_mask": torch.ones((1, 1), device=native.device, dtype=torch.bool),
        "text_phrase_weight": torch.tensor([[get_phrase_type_weight(classify_phrase(description))]], device=native.device),
        "text_dependency_adj": torch.ones((1, 1, 1), device=native.device),
    }
    with torch.inference_mode():
        raw, maps, _, _ = _run_text_inference(native.backend, {"img": tensor.to(native.device), **conditions})
        # Original NMS mutates xywh in place; compare the pre-NMS raw forward
        # output captured by the native hook, rather than the mutated tensor.
        raw_before_nms = _first_tensor(raw).detach().cpu().clone()
        detections = non_max_suppression(raw, conf_thres=.25, iou_thres=.7, multi_label=False, agnostic=True, max_det=300)[0]
        selected = None
        phrase_scores = torch.empty((1, 0), device=native.device, dtype=tensor.dtype)
        if match and len(detections):
            phrase_scores = score_matrix_from_phrase_maps(
                phrase_logits_levels=maps, image_index=0,
                token_indices=torch.zeros(1, device=native.device, dtype=torch.long),
                candidate_boxes_xyxy=detections[:, :4], input_hw=tuple(tensor.shape[2:]),
            )
            rows, columns = assign_by_score_matrix(phrase_scores, temperature=.7)
            _require(rows.numel() == columns.numel() == 1 and int(rows[0]) == 0, "reference_grounding_assignment")
            selected = int(columns[0])
        network_boxes = detections[:, :4].clone()
        boxes = network_boxes.clone()
        if len(boxes):
            scale_boxes(tensor.shape[2:], boxes, original_hw, ratio_pad=ratio_pad)
        # Keep the original evaluator's legacy output above, but independently
        # invert its real per-axis resize metadata for the API geometry check.
        (gain_h, gain_w), (left, top) = ratio_pad
        height, width = original_hw
        geometry_boxes = np.clip(
            (network_boxes.detach().cpu().numpy().astype(np.float64)
             - np.array([left, top, left, top], dtype=np.float64))
            / np.array([gain_w, gain_h, gain_w, gain_h], dtype=np.float64),
            0, [width, height, width, height],
        )
    return {
        "raw": raw_before_nms,
        "maps": [value.detach().cpu().clone() for value in maps],
        "boxes": boxes.detach().cpu(), "scores": detections[:, 4].detach().cpu(),
        "network_boxes": network_boxes.detach().cpu(), "geometry_boxes": geometry_boxes,
        "phrase_scores": phrase_scores.detach().cpu(),
        "selected": selected, "conditions": conditions,
    }


def _grounding_selection_parity(native: Any, description: str, prediction: Any,
                                reference: dict[str, Any]) -> dict[str, Any]:
    """Check deployment selection with an independent NumPy spatial formula.

    The original phrase-map matcher remains the semantic reference. An enabled
    absolute-position bonus may change its index, without changing the model,
    NMS candidates or their confidence scores. No annotations enter this check.
    """
    from grounding_spatial import parse_absolute_position

    semantic = torch.as_tensor(reference["phrase_scores"]).detach().cpu().numpy()
    # The separate geometry check verifies these native boxes against the
    # independent resize inverse. Use the deployed dtype here so a float64
    # reference inverse cannot flip an otherwise exact float32 score tie.
    boxes = np.asarray(prediction.boxes, dtype=semantic.dtype)
    _require(semantic.shape == (1, len(boxes)) and np.isfinite(semantic).all(),
             "detection_semantic_score_shape")
    weight = float(getattr(native, "spatial_weight", 0.0))
    _require(np.isfinite(weight) and 0 <= weight <= 1, "detection_spatial_weight")
    hint = parse_absolute_position(description) if weight else None
    final = semantic.copy()
    if hint is not None and len(boxes):
        height, width = prediction.original_size
        axes = []
        for direction, first, second, extent in (
            (hint.horizontal, 0, 2, width), (hint.vertical, 1, 3, height),
        ):
            if direction is None:
                continue
            coordinate = np.clip((boxes[:, first] + boxes[:, second]) / 2 / extent, 0, 1)
            if direction == -1:
                axes.append(1 - coordinate)
            elif direction == 1:
                axes.append(coordinate)
            elif direction == 0:
                axes.append(1 - 2 * np.abs(coordinate - .5))
            else:
                raise AssertionError("Native parity contract mismatch: detection_spatial_hint")
        _require(bool(axes), "detection_spatial_hint_axes")
        bonus = np.mean(np.stack(axes), axis=0)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        bonus = np.where(valid, bonus, 0)
        final = final + weight * bonus[None, :]
    expected = int(np.argmax(final[0])) if final.shape[1] else None
    original = reference["selected"]
    if hint is None:
        _require(expected == original, "detection_original_phrase_assignment")
    _require(prediction.grounding_index == expected, "detection_spatial_selection")
    return {
        "original_selected_index": original, "expected_selected_index": expected,
        "native_selected_index": prediction.grounding_index,
        "spatial_weight": weight,
        "spatial_hint": {"horizontal": hint.horizontal, "vertical": hint.vertical} if hint is not None else None,
        "assignment_equal": prediction.grounding_index == original,
        "selection_policy_checked": True,
    }


def _detection_semantic_score_parity(observed: dict[str, Any], reference: dict[str, Any],
                                    input_hw: tuple[int, int]) -> float:
    """Compare original ROI scores using the captured native phrase maps."""
    from text_encoder.matching import score_matrix_from_phrase_maps

    scores = score_matrix_from_phrase_maps(
        phrase_logits_levels=observed["phrase_maps"], image_index=0,
        token_indices=torch.zeros(1, dtype=torch.long),
        candidate_boxes_xyxy=reference["network_boxes"], input_hw=input_hw,
    )
    return _close("detection_semantic_phrase_scores", scores, reference["phrase_scores"])


def _detection(adapter: Any, image: Path) -> dict[str, Any]:
    from hfsa_adapter import QuestionPlan

    native = adapter.detection
    _require(not native.model.training, "detection_eval")
    stride = int(torch.as_tensor(native.backend.stride).max())
    decoded = cv2.imread(str(image))
    _require(decoded is not None, "detection_reference_decode")
    image_size = native.image_size_for(decoded.shape[:2])
    tensor, ratio_pad, original_hw = _original_detection_image(image, stride, image_size)
    description = "The gray small windmill"
    reference = _detection_reference(native, tensor, description, ratio_pad, original_hw, match=True)
    question = "Return the bounding box. Description: " + description
    with _capture_forward(native.model) as observed:
        result = native.predict(image, question)
    input_error = _close("detection_rect_input", observed["image"], tensor)
    feature_error = _close("detection_forward_features", observed["conditions"]["txt_vec"], reference["conditions"]["txt_vec"])
    for native_key, reference_key in (
        ("txt_token_mask", "text_token_mask"), ("txt_phrase_weight", "text_phrase_weight"),
        ("txt_dependency_adj", "text_dependency_adj"),
    ):
        _close("detection_forward_conditions", observed["conditions"][native_key], reference["conditions"][reference_key], atol=0, rtol=0)
    raw_error = _close("detection_raw_predictions", observed["raw"], reference["raw"], atol=2e-3)
    boxes_error = _close("detection_per_axis_boxes", result.boxes, reference["geometry_boxes"], atol=2e-3)
    legacy_boxes_error = float(np.max(np.abs(
        result.boxes.astype(np.float64) - reference["boxes"].numpy().astype(np.float64),
    ))) if result.boxes.size else 0.0
    scores_error = _close("detection_scores", result.scores, reference["scores"])
    selection = _grounding_selection_parity(native, description, result, reference)
    _require(len(observed["phrase_maps"]) == len(reference["maps"]), "detection_phrase_levels")
    map_error = max((_close("detection_phrase_maps", actual, expected, atol=1e-5)
                     for actual, expected in zip(observed["phrase_maps"], reference["maps"])), default=0.0)
    semantic_error = _detection_semantic_score_parity(observed, reference, tuple(tensor.shape[2:]))
    selected = selection["expected_selected_index"]
    expected_box = None
    empty_code = "empty_detections"
    height, width = original_hw
    if selected is not None:
        box = reference["geometry_boxes"][selected]
        if box[2] > box[0] and box[3] > box[1]:
            expected_box = [int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3]))]
        else:
            empty_code = "empty_detection_box"
    with _backend(adapter, "detection"):
        _api_box_check(adapter, image, question, expected_box, empty_code, height, width)

    presence_reference = _detection_reference(native, tensor, "windmill", ratio_pad, original_hw, match=False)
    presence_question = "Is there a windmill in the image?"
    with _capture_forward(native.model) as presence_observed:
        presence_result = native.predict(image, presence_question, require_grounding=False)
    _close("presence_input", presence_observed["image"], tensor)
    _close("presence_forward_features", presence_observed["conditions"]["txt_vec"], presence_reference["conditions"]["txt_vec"])
    _close("presence_raw_predictions", presence_observed["raw"], presence_reference["raw"], atol=2e-3)
    _close("presence_per_axis_boxes", presence_result.boxes, presence_reference["geometry_boxes"], atol=2e-3)
    _close("presence_scores", presence_result.scores, presence_reference["scores"])
    _require(presence_result.grounding_index is None, "presence_no_grounding_assignment")
    _require(presence_result.detection_count == len(presence_reference["boxes"]), "presence_detection_count")
    mapped = adapter._predict_plan(
        _request(presence_question, "enum", height, width, values=["Yes", "No"]),
        image, QuestionPlan("presence", presence_question),
    )
    _require(mapped == ("Yes" if len(presence_reference["boxes"]) else "No"), "presence_boolean_mapping")
    return {
        "checked": True, "candidate_count": len(reference["boxes"]),
        "grounding_match_checked": selected is not None, "input_hw": list(tensor.shape[2:]),
        "input_max_abs_error": input_error, "raw_max_abs_error": raw_error,
        "text_feature_max_abs_error": feature_error,
        "boxes_max_abs_error": boxes_error, "scores_max_abs_error": scores_error,
        "coordinate_restore": "per_axis_resize_gain",
        "legacy_boxes_max_abs_difference": legacy_boxes_error,
        "phrase_maps_max_abs_error": map_error, "api_mapping_checked": True,
        "semantic_scores_max_abs_error": semantic_error,
        "selection": selection,
        "presence_checked": True, "presence_api_boolean_mapping_checked": True,
        "presence_candidate_count": len(presence_reference["boxes"]),
    }


def _refseg(adapter: Any, image: Path, rgb: Image.Image, native: Any) -> dict[str, Any]:
    from dataset.rrsisd_refseg_dataset import RRSISDRefSegDataset
    from tasks.refseg.runtime import predict_with_optional_text

    _require(not native.model.training and native.imgsz == 512, "refseg_original_eval_configuration")
    description = "The gray small windmill"
    original = RRSISDRefSegDataset._read_image(str(image))
    height, width = original.shape[:2]
    # A synthetic zero array satisfies the static resize signature only. It is
    # neither ground truth nor a model input, and no dataset instance is built.
    resized, _ = RRSISDRefSegDataset._resize(original, np.zeros((height, width), dtype=np.uint8), 512)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255
    features, token_mask = native.text_encoder.encode([description], batch_size=1, return_tokens=True)
    with torch.inference_mode():
        reference = predict_with_optional_text(
            native.model, {"img": tensor.to(native.device),
                           "text_embedding": features.to(device=native.device, dtype=torch.float32),
                           "text_token_mask": token_mask.to(device=native.device, dtype=torch.bool)}, True,
        )
        logits = _first_tensor(reference)
        probability = F.interpolate(logits.float().sigmoid(), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    reference_probability = probability.detach().cpu()
    reference_mask = reference_probability.numpy() > native.threshold
    question = "Return the bounding box. Description: " + description
    with _capture_forward(native.model) as observed:
        result = native.predict(image, question)
    input_error = _close("refseg_input", observed["image"], tensor)
    _close("refseg_forward_features", observed["conditions"]["text_embedding"], features)
    _close("refseg_forward_token_mask", observed["conditions"]["text_token_mask"], token_mask, atol=0, rtol=0)
    logits_error = _close("refseg_logits", observed["raw"], logits)
    probability_error = _close("refseg_native_probability", result.probability, reference_probability)
    _require(np.array_equal(result.mask, reference_mask), "refseg_validation_threshold_mask")
    _require(result.threshold == native.threshold, "refseg_checkpoint_threshold")
    ys, xs = reference_mask.nonzero()
    expected_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    with _backend(adapter, "refseg", refseg=native):
        _api_box_check(adapter, image, question, expected_box, "empty_refseg_mask", height, width)
    return {
        "checked": True, "input_hw": [512, 512], "input_max_abs_error": input_error,
        "logits_max_abs_error": logits_error, "probability_max_abs_error": probability_error,
        "binary_mask_equal": True, "nonempty_mask_checked": bool(len(xs)), "api_mapping_checked": True,
    }


def verify_native_parity(adapter: Any, image: str | Path, refseg: Any = None) -> dict[str, Any]:
    """Check three production models, or only a supplied existing/temporary RefSeg.

    Call once before listening. A later refseg= call checks only that already
    loaded model so a default three-model server never retains RefSeg or another
    CLIP instance. All temporary hooks and adapter backend changes are restored.
    """
    image = Path(image).expanduser().resolve()
    with Image.open(image) as source:
        source.load()
        rgb = source.convert("RGB")
    if refseg is not None:
        report = {"refseg": _refseg(adapter, image, rgb, refseg)}
    else:
        report = {
            "classification": _classification(adapter, image, rgb),
            "counting": _counting(adapter, image, rgb),
            "detection": _detection(adapter, image),
        }
    print(json.dumps({"native_parity": report}, ensure_ascii=True), flush=True)
    return report
