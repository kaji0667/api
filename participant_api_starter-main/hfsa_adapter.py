"""Official protocol adapter; native task runtimes own model inference.

A local Qwen task selector routes questions independently of the manual Web
router. Unsupported visual tasks fail explicitly, without fabricated answers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_task_text import TaskInputError, count_option_matches, detection_text
from ollama_router import TASKS, TASK_FORMATS, RouterError, ensure_router_ready, route_question, validate_plan


# Aliases describe the actual 21-class checkpoint, not arbitrary scene classes.
SCENE_ALIASES = {
    "airport": ("airport", "机场", "飞机场"),
    "baseball-diamond": ("baseball diamond", "baseball field", "棒球场"),
    "basketball-court": ("basketball court", "篮球场"),
    "bridge": ("bridge", "桥", "桥梁", "大桥"),
    "chimney": ("chimney", "烟囱"),
    "dam": ("dam", "大坝", "水坝"),
    "expressway-service-area": ("expressway service area", "highway service area", "高速公路服务区"),
    "expressway-toll-station": ("expressway toll station", "highway toll station", "高速公路收费站"),
    "golffield": ("golffield", "golf field", "golf course", "高尔夫球场"),
    "ground-track-field": ("ground track field", "running track", "track and field", "田径场", "操场"),
    "harbor": ("harbor", "harbour", "port", "港口", "港湾"),
    "helipad": ("helipad", "直升机停机坪"),
    "overpass": ("overpass", "立交桥"),
    "roundabout": ("roundabout", "环岛"),
    "soccer-ball-field": ("soccer ball field", "soccer field", "football field", "足球场"),
    "stadium": ("stadium", "体育场"),
    "storage-tank": ("storage tank", "oil storage tank", "储罐", "储油罐"),
    "swimming-pool": ("swimming pool", "游泳池"),
    "tennis-court": ("tennis court", "网球场"),
    "trainstation": ("trainstation", "train station", "railway station", "火车站"),
    "windmill": ("windmill", "wind turbine", "风车", "风力发电机"),
}


def normalize_label(value: str) -> str:
    value = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
    return re.sub(r"^(?:a|an|the)\s+", "", value).strip(" 。.?!？！")


_SCENE_LOOKUP = {
    normalize_label(alias): label
    for label, aliases in SCENE_ALIASES.items()
    for alias in (label, *aliases)
}
@dataclass(frozen=True)
class QuestionPlan:
    task: str
    original_question: str


class AdapterFailure(ValueError):
    """Expected adapter rejection with a fixed, safe diagnostic code."""

    def __init__(self, code: str, message: str, *, task: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.task = task if task in TASKS else None


_DIAGNOSTIC_LOCK = threading.Lock()
# Codes are part of the local diagnostic contract, never arbitrary exception data.
_DIAGNOSTIC_REASONS = frozenset({
    "task_constraint_mismatch", "unsupported_task", "unsupported_image_count",
    "router_invalid_constraint", "router_invalid_image_count", "router_no_compatible_executor",
    "router_invalid_output", "router_empty_question", "router_input_too_long",
    "router_requires_loopback", "router_invalid_timeout", "router_response_too_large",
    "router_incomplete_response", "router_unavailable_or_timeout", "router_warmup_task_mismatch",
    "unsupported_scene_options", "ambiguous_scene_options", "count_below_minimum",
    "count_option_unmatched", "unsupported_presence_options", "ambiguous_presence_options",
    "empty_task_text", "ambiguous_description", "unsupported_count_query",
    "unsupported_text_tokenizer", "missing_target", "text_context_exceeded",
    "invalid_text_embedding", "unsupported_detection_backend", "invalid_detection_image",
    "detection_phrase_maps_missing", "invalid_detection_phrase_maps", "invalid_detection_geometry",
    "invalid_detection_matching", "invalid_detection_result", "image_dimensions_mismatch",
    "empty_detections", "invalid_grounding_match", "empty_detection_box", "empty_refseg_mask",
    "invalid_refseg_checkpoint", "invalid_refseg_configuration", "invalid_refseg_text_encoder",
    "invalid_refseg_image", "invalid_refseg_text_features", "invalid_refseg_result",
    "vqa_not_configured", "vqa_invalid_options", "vqa_invalid_input", "invalid_vqa_result",
    "vqa_answer_contract", "vqa_unsupported_language", "vqa_unsupported_task",
    "vqa_choice_unmatched", "vqa_text_context_exceeded", "vqa_nonfinite_prediction",
    "grounding_hybrid_selected", "grounding_unsupported_plan", "grounding_empty_candidates",
    "grounding_missing_scores", "grounding_crop_failed", "grounding_rank_failed",
    "grounding_invalid_selection",
    "grounding_region_selected", "grounding_reference_unavailable",
    "grounding_conflicting_region", "grounding_roi_unavailable",
    "grounding_container_unavailable", "grounding_no_spatial_constraints",
    "grounding_parser_invalid_output", "grounding_parser_source_mismatch",
    "grounding_parser_unresolved", "grounding_parser_conflicting_position",
    "grounding_parser_unaccounted_text", "grounding_parser_input_too_long",
    "grounding_parser_unavailable",
})
_DIAGNOSTIC_ERROR_TYPES = frozenset({
    "AdapterFailure", "ValueError", "TypeError", "KeyError", "RuntimeError", "OSError",
    "FileNotFoundError", "PermissionError", "TimeoutError", "MemoryError",
})


def _append_prediction_diagnostic(encoded: str) -> None:
    """Best-effort private JSONL; log failures never change inference results."""
    try:
        filename = os.environ.get("HFSA_API_DIAGNOSTIC_FILE", "").strip()
        if not filename:
            return
        path = Path(filename).expanduser()
        with _DIAGNOSTIC_LOCK:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = (os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
            fd = os.open(path, flags, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    return
                handle.write(encoded + "\n")
    except Exception:
        # Do not print paths, exception text or a traceback from logging itself.
        pass


def _capture_replay_request(request: dict[str, Any]) -> None:
    """Optional private request copy for local replay, excluding credentials/gold."""
    directory = os.environ.get("HFSA_API_REPLAY_DIR", "").strip()
    if not directory:
        return
    try:
        fields = ("protocol_version", "request_id", "item_id", "question", "answer_type",
                  "choices", "image_width", "image_height")
        payload = {name: request[name] for name in fields if name in request}
        payload["images"] = [
            {name: spec[name] for name in ("asset_id", "sha256", "mime_type") if name in spec}
            for spec in request.get("images", [])
        ]
        fields = ("type", "values", "case_insensitive", "question_form", "coordinate_format",
                  "length", "min_length", "minimum")
        payload["response_constraint"] = {
            name: request["response_constraint"][name]
            for name in fields if name in request.get("response_constraint", {})
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
        if len(encoded) > 128 * 1024:
            raise ValueError("Replay request exceeds protocol size limit")
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        path = target / (hashlib.sha256(encoded).hexdigest() + ".json")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
    except Exception as error:
        # Capture errors must not fail inference or reveal request/exception text.
        try:
            print(json.dumps({"event": "hfsa_api_replay_capture", "status": "failed",
                              "error_type": type(error).__name__}), file=sys.stderr, flush=True)
        except OSError:
            pass


def _prediction_diagnostic(request: dict[str, Any], image_count: int,
                           plan: QuestionPlan | None, started: float,
                           error: Exception | None = None,
                           execution_metadata: dict[str, Any] | None = None) -> None:
    # Never log question/choices/answers, image paths, headers or exception text.
    def identifier(value: Any) -> str | None:
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            return value
        return None

    kind = request.get("response_constraint", {}).get("type")
    task = plan.task if plan is not None else getattr(error, "task", None)
    task = task if isinstance(task, str) and task in TASKS else None
    record = {
        "event": "hfsa_api_prediction",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "process_id": os.getpid(),
        "item_id": identifier(request.get("item_id")),
        "request_id": identifier(request.get("request_id")),
        "constraint_type": kind if kind in {"enum", "single_choice", "short_text", "integer", "bbox"} else None,
        "image_count": image_count,
        "task": task,
        "executor": None,
        "status": "failed" if error is not None else "succeeded",
        "reason": None,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }
    if error is not None:
        code = error.code if isinstance(error, AdapterFailure) else None
        record["reason"] = code if isinstance(code, str) and code in _DIAGNOSTIC_REASONS else "unexpected_exception"
        name = type(error).__name__
        record["error_type"] = name if name in _DIAGNOSTIC_ERROR_TYPES else "Exception"
    if execution_metadata:
        executor = execution_metadata.get("executor")
        if isinstance(executor, str) and executor in {"classification", "counting", "detection", "refseg", "vqa"}:
            record["executor"] = executor
    if task in {"presence", "visual_grounding"} and execution_metadata:
        diagnostic = execution_metadata.get("detection_inference")
        if isinstance(diagnostic, dict):
            # Copy a numeric allowlist only, never model prompts/boxes/answers.
            safe = {}
            for key in ("original_hw", "input_hw"):
                value = diagnostic.get(key)
                if (isinstance(value, (list, tuple)) and len(value) == 2
                        and all(type(n) is int and n > 0 for n in value)):
                    safe[key] = list(value)
            for key in ("confidence_threshold", "max_confidence"):
                value = diagnostic.get(key)
                if type(value) in {int, float} and 0 <= value <= 1:
                    safe[key] = round(float(value), 8)
            for key in ("raw_anchor_count", "pre_nms_above_threshold", "nms_candidate_count"):
                value = diagnostic.get(key)
                if type(value) is int and value >= 0:
                    safe[key] = value
            for key, allowed in (
                ("grounding_policy", {"native", "spatial_reference"}),
                ("grounding_parse_status", {"parsed", "unsupported"}),
                ("grounding_parser", {"qwen", "rules"}),
                ("grounding_outcome", {"region_selected", "native_fallback"}),
                ("grounding_reason", _DIAGNOSTIC_REASONS),
            ):
                value = diagnostic.get(key)
                if isinstance(value, str) and value in allowed:
                    safe[key] = value
            for key in ("grounding_reference_count", "grounding_candidate_count",
                        "grounding_located_reference_count", "grounding_roi_count",
                        "grounding_detector_call_count", "grounding_elapsed_ms",
                        "grounding_container_count", "grounding_located_container_count"):
                value = diagnostic.get(key)
                if type(value) is int and value >= 0:
                    safe[key] = value
            if safe:
                record["detection_inference"] = safe
    encoded = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    try:
        print(encoded, file=sys.stderr, flush=True)
    except Exception:
        # An unavailable log destination must not change the prediction result.
        pass
    _append_prediction_diagnostic(encoded)


def parse_question(request: dict[str, Any]) -> QuestionPlan:
    try:
        value = validate_plan(route_question(request))
    except RouterError as error:
        raise AdapterFailure(error.code, "Local question router failed", task=error.task) from None
    task = value["task"]
    kind = request["response_constraint"]["type"]
    if task in TASK_FORMATS and kind not in TASK_FORMATS[task]:
        raise AdapterFailure("task_constraint_mismatch", "Task and answer contract differ", task=task)
    # Select an executor only; preserve the complete source question verbatim.
    return QuestionPlan(task, request["question"])


def _constrained_candidates(request: dict[str, Any]) -> list[tuple[str, str]]:
    constraint = request["response_constraint"]
    if constraint["type"] == "single_choice":
        return [(key, request["choices"][key]) for key in constraint["values"]]
    return [(value, value) for value in constraint["values"]]


def validate_api_vqa_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Serving requires the current architecture and a completed full-run evaluation."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("architecture_id") != "hfsa_frozen_qformer_vqa_v2"
            or manifest.get("fusion_backend") != "blip2_qformer"
            or manifest.get("evaluation_status") != "complete"):
        raise ValueError("API requires a fully evaluated Q-Former VQA bundle")
    receipts = {}
    for name, relative in (("training", "training_result.json"),
                           ("config", "config/train.json"),
                           ("metrics", "evaluation/metrics.json")):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Unsafe VQA receipt path")
        data = path.read_bytes()
        digest = manifest.get("file_sha256", {}).get(relative)
        if not isinstance(digest, str) or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Missing/mismatched full VQA training receipt")
        receipts[name] = json.loads(data)
    training, config, metrics = (receipts[key] for key in ("training", "config", "metrics"))
    count = training.get("evaluated_val_qa")
    checkpoint_hash = manifest.get("checkpoint", {}).get("sha256")
    dataset_hash = config.get("val_jsonl_sha256")
    for value in (checkpoint_hash, dataset_hash):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("API requires explicit checkpoint/dataset SHA-256")
    best_epoch = training.get("best_epoch")
    macro = metrics.get("first_stage_macro_accuracy")
    best_macro = training.get("best_macro_accuracy")
    if not all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
               for value in (macro, best_macro)) or abs(macro - best_macro) > 1e-10:
        raise ValueError("API requires matching finite full-validation metrics")
    if (training.get("validation_scope") != "full_split"
            or config.get("fusion_backend") != "blip2_qformer"
            or config.get("max_train_batches") != 0 or config.get("max_val_batches") != 0
            or type(count) is not int or count <= 0 or metrics.get("n") != count
            or type(training.get("epochs_completed")) is not int or training["epochs_completed"] <= 0
            or type(best_epoch) is not int or not 0 < best_epoch <= training["epochs_completed"]
            or metrics.get("checkpoint_sha256") != manifest.get("checkpoint", {}).get("sha256")
            or metrics.get("dataset_sha256") != config.get("val_jsonl_sha256")):
        raise ValueError("API rejects incomplete, limited or mismatched VQA training")
    return {"architecture_id": manifest["architecture_id"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "epochs_completed": training["epochs_completed"], "best_epoch": training["best_epoch"],
            "validation_scope": "full_split", "validation_qa": count}


class HFSAAdapter:
    def __init__(self, classification: Any, detection: Any, counting: Any, refseg: Any,
                 *, grounding_backend: str = "detection", vqa: Any = None,
                 grounding_policy: str = "native", grounding_executor: Any = None) -> None:
        if grounding_backend not in {"detection", "refseg"}:
            raise ValueError("Grounding backend must be detection or refseg")
        if grounding_policy not in {"native", "spatial_reference"}:
            raise ValueError("Grounding policy must be native or spatial_reference")
        if grounding_policy == "spatial_reference" and grounding_backend != "detection":
            raise ValueError("spatial_reference requires the detection grounding backend")
        if (grounding_policy == "spatial_reference") != (grounding_executor is not None):
            raise ValueError("spatial_reference requires an initialized bbox task executor")
        self.classification = classification
        self.detection = detection
        self.counting = counting
        self.refseg = refseg
        self.vqa = vqa
        self.grounding_backend = grounding_backend
        self.grounding_policy = grounding_policy
        self.grounding_executor = grounding_executor

    @classmethod
    def from_local(cls) -> HFSAAdapter:
        grounding_backend = os.environ.get("HFSA_GROUNDING_BACKEND", "detection").strip().lower()
        if grounding_backend not in {"detection", "refseg"}:
            raise ValueError("HFSA_GROUNDING_BACKEND must be detection or refseg")
        default_policy = "spatial_reference" if grounding_backend == "detection" else "native"
        grounding_policy = os.environ.get("HFSA_GROUNDING_POLICY", default_policy).strip().lower()
        if grounding_policy not in {"native", "spatial_reference"}:
            raise ValueError("HFSA_GROUNDING_POLICY must be native or spatial_reference")
        if grounding_policy == "spatial_reference" and grounding_backend != "detection":
            raise ValueError("spatial_reference requires HFSA_GROUNDING_BACKEND=detection")
        max_detection_size = int(os.environ.get("HFSA_DETECTION_MAX_IMGSZ", "1600"))
        if max_detection_size < 800 or max_detection_size % 32:
            raise ValueError("HFSA_DETECTION_MAX_IMGSZ must be a multiple of 32, at least 800")
        spatial_weight = float(os.environ.get("HFSA_GROUNDING_SPATIAL_WEIGHT", "0.15"))
        if not math.isfinite(spatial_weight) or not 0 <= spatial_weight <= 1:
            raise ValueError("HFSA_GROUNDING_SPATIAL_WEIGHT must be a finite number between 0 and 1")
        print("HFSA API: checking local Qwen text router", flush=True)
        ensure_router_ready()
        code_dir = Path(__file__).resolve().parents[1]
        weights_dir = Path(os.environ.get("HFSA_MODELS_DIR", str(code_dir.parent / "HFSA_models"))).expanduser().resolve()
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from native_classification import NativeClassification
        from native_counting import NativeCounting
        from native_detection import NativeDetection, NativeTextEncoder
        from tasks.vqa.inference import VQAPredictor

        device = os.environ.get("HFSA_API_DEVICE", "auto").strip() or "auto"
        # Keep the additional frozen VQA encoders off the 8-GiB deployment GPU.
        vqa_device = os.environ.get("HFSA_VQA_DEVICE", "cpu").strip() or "cpu"
        vqa_bundle = Path(os.environ.get("HFSA_VQA_BUNDLE_DIR", str(code_dir / "runs/vqa/qformer"))).expanduser().resolve()
        checkpoints = {
            "classification": weights_dir / "runs/classification/vrsbench_scene/weights/best.pt",
            "detection": weights_dir / "runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt",
            "counting": weights_dir / "runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt",
            "pretrained": weights_dir / "pretrain_model/yolov12m.pt",
            "vqa_text": weights_dir / "openclip/ViT-L-14.pt",
            "vqa_manifest": vqa_bundle / "manifest.json",
        }
        if grounding_backend == "refseg":
            checkpoints["refseg"] = weights_dir / "runs/semseg/srp_yolov12m_axis/weights/best_raw.pt"
        for name, path in checkpoints.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} weights: {path}")

        validate_api_vqa_bundle(vqa_bundle)

        print("HFSA API: loading scene classification", flush=True)
        classification = NativeClassification(
            checkpoint=checkpoints["classification"], pretrained_weights=checkpoints["pretrained"],
            model_yaml=str(code_dir / "ultralytics/cfg/models/v12/yolov12m-classification.yaml"), device=device,
        )
        print("HFSA API: loading detection and shared text encoder", flush=True)
        shared_encoder = NativeTextEncoder(device=device)
        detection = NativeDetection(
            checkpoint=checkpoints["detection"], device=device, encoder=shared_encoder,
            max_image_size=max_detection_size,
            spatial_weight=spatial_weight, candidate_confidence=.25,
        )
        grounding_executor = None
        if grounding_policy == "spatial_reference":
            from tasks.detection.spatial_reference_grounding import SpatialReferenceGroundingExecutor
            from tasks.detection.structured_spatial_parser import QwenSpatialPlanParser

            grounding_executor = SpatialReferenceGroundingExecutor(
                detection, description_extractor=detection_text, image_converter=detection._image,
                plan_parser=QwenSpatialPlanParser(),
            )
        print("HFSA API: loading counting", flush=True)
        counting = NativeCounting(
            checkpoint=checkpoints["counting"], device=device, encoder=shared_encoder,
        )
        print(f"HFSA API: loading trained Q-Former VQA on {vqa_device}", flush=True)
        vqa = VQAPredictor(
            bundle_dir=vqa_bundle, device=vqa_device,
            asset_paths={"visual_backbone": checkpoints["pretrained"],
                         "text_encoder": checkpoints["vqa_text"]},
        )
        refseg = None
        if grounding_backend == "refseg":
            from native_refseg import NativeRefSegPredictor

            print("HFSA API: loading referring segmentation", flush=True)
            refseg = NativeRefSegPredictor(
                checkpoint=checkpoints["refseg"], device=device,
                model_yaml=str(code_dir / "ultralytics/cfg/models/v12/yolov12m-semseg.yaml"),
            )
        print(f"HFSA API: bbox grounding backend={grounding_backend} policy={grounding_policy}", flush=True)
        adapter = cls(classification, detection, counting, refseg,
                      grounding_backend=grounding_backend, vqa=vqa,
                      grounding_policy=grounding_policy, grounding_executor=grounding_executor)
        # Explicit local validation reuses the models about to serve requests.
        # It never loads a duplicate detector/CLIP or uses scoring annotations.
        if os.environ.get("HFSA_API_VERIFY_PARITY") == "1":
            from native_parity import verify_native_parity
            parity_image = Path(os.environ["HFSA_API_PARITY_IMAGE"])
            verify_native_parity(adapter, parity_image)
            if refseg is not None:
                verify_native_parity(adapter, parity_image, refseg=refseg)
            else:
                import torch
                from native_refseg import NativeRefSegPredictor

                def existing_token_encoder(**kwargs: Any) -> Any:
                    current = shared_encoder.encoder
                    if (kwargs["model_name"] != current.model_name
                            or kwargs["pretrained"] != current.pretrained
                            or kwargs["device"] != str(current.device)
                            or kwargs["precision"] != "fp32" or not kwargs["normalize"]):
                        raise ValueError("RefSeg parity encoder configuration differs")
                    return current

                temporary_refseg = NativeRefSegPredictor(
                    checkpoint=weights_dir / "runs/semseg/srp_yolov12m_axis/weights/best_raw.pt",
                    device=device,
                    model_yaml=str(code_dir / "ultralytics/cfg/models/v12/yolov12m-semseg.yaml"),
                    text_encoder_factory=existing_token_encoder,
                )
                verify_native_parity(adapter, parity_image, refseg=temporary_refseg)
                del temporary_refseg
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return adapter

    def _model_text(self, text: str) -> str:
        # Validation does not change the source question, including whitespace.
        if not isinstance(text, str) or not text.strip():
            raise AdapterFailure("router_empty_question", "Missing source question")
        return text

    def _scene(self, request: dict[str, Any], image: Path,
               execution_metadata: dict[str, Any] | None = None) -> str:
        kind = request["response_constraint"]["type"]
        if kind == "short_text":
            if execution_metadata is not None:
                execution_metadata["executor"] = "classification"
            return str(self.classification.predict(image, topk=1)["predictions"][0]["class_name"])
        candidates = _constrained_candidates(request)
        mapped = [(key, _SCENE_LOOKUP.get(normalize_label(text))) for key, text in candidates]
        available = set(self.classification.class_names)
        if any(label is None or label not in available for _, label in mapped):
            raise AdapterFailure("unsupported_scene_options", "Scene options include a class unsupported by the checkpoint")
        if len({label for _, label in mapped}) != len(mapped):
            raise AdapterFailure("ambiguous_scene_options", "Scene options map ambiguously to the same class")
        if execution_metadata is not None:
            execution_metadata["executor"] = "classification"
        predictions = self.classification.predict(image, topk=len(available))["predictions"]
        scores = {row["class_name"]: row["probability"] for row in predictions}
        return max(mapped, key=lambda item: scores[item[1]])[0]

    def _vqa(self, request: dict[str, Any], image: Path, question: str,
             execution_metadata: dict[str, Any] | None = None) -> str:
        if self.vqa is None:
            raise AdapterFailure("vqa_not_configured", "Trained VQA executor is not loaded")
        kind = request["response_constraint"]["type"]
        kwargs: dict[str, Any] = {}
        if kind in {"single_choice", "enum"}:
            try:
                candidates = _constrained_candidates(request)
            except (KeyError, TypeError):
                raise AdapterFailure("vqa_invalid_options", "Invalid VQA answer options") from None
            if (not candidates or any(not isinstance(key, str) or not key.strip()
                           or not isinstance(value, str) or not value.strip()
                           for key, value in candidates)
                    or len({key for key, _ in candidates}) != len(candidates)):
                raise AdapterFailure("vqa_invalid_options", "Invalid VQA answer options")
            if kind == "single_choice":
                kwargs["choices"] = dict(candidates)
            else:
                kwargs["allowed_answers"] = [key for key, _ in candidates]
        elif kind != "short_text":
            raise AdapterFailure("task_constraint_mismatch", "VQA requires a text answer")
        if execution_metadata is not None:
            execution_metadata["executor"] = "vqa"
        try:
            result = self.vqa.predict(image, question, **kwargs)
        except ValueError:
            raise AdapterFailure("vqa_invalid_input", "VQA input cannot be processed") from None
        if not isinstance(result, dict):
            raise AdapterFailure("invalid_vqa_result", "Invalid VQA output")
        if result.get("status") != "ok":
            reason = result.get("reason")
            if (result.get("status") not in {"unsupported", "error"}
                    or not isinstance(reason, str)
                    or reason not in {"unsupported_language", "unsupported_task", "choice_unmatched",
                              "text_context_exceeded", "nonfinite_prediction"}):
                raise AdapterFailure("invalid_vqa_result", "Invalid VQA failure output")
            raise AdapterFailure("vqa_" + reason, "Trained VQA cannot answer this request")
        answer = result.get("answer")
        from tasks.vqa.config import SUPPORTED

        if (result.get("question_type") not in SUPPORTED
                or not isinstance(answer, str) or not answer.strip() or len(answer) > 4096):
            raise AdapterFailure("invalid_vqa_result", "Invalid VQA answer")
        if kind in {"single_choice", "enum"}:
            if answer not in request["response_constraint"]["values"]:
                raise AdapterFailure("vqa_answer_contract", "VQA answer is outside allowed values")
        elif len(answer) < request["response_constraint"].get("min_length", 1):
            raise AdapterFailure("vqa_answer_contract", "VQA answer is shorter than required")
        return answer

    @staticmethod
    def _number(request: dict[str, Any], count: int) -> str:
        answer = str(count)
        constraint = request["response_constraint"]
        if constraint["type"] in {"integer", "short_text"}:
            if count < constraint.get("minimum", 0):
                raise AdapterFailure("count_below_minimum", "Predicted count violates the answer minimum")
            return answer
        matches = []
        for key, value in _constrained_candidates(request):
            if count_option_matches(value, count):
                matches.append(key)
        if len(matches) != 1:
            raise AdapterFailure("count_option_unmatched", "Predicted count has no unique matching answer option")
        return matches[0]

    @staticmethod
    def _presence(request: dict[str, Any], found: bool) -> str:
        if request["response_constraint"]["type"] == "short_text":
            return "Yes" if found else "No"
        yes, no = {"yes", "true", "是", "有", "存在"}, {"no", "false", "否", "无", "没有", "不存在"}
        candidates = _constrained_candidates(request)
        if any(normalize_label(text) not in yes | no for _, text in candidates):
            raise AdapterFailure("unsupported_presence_options", "Presence options must express Yes/No")
        matches = [key for key, text in candidates if normalize_label(text) in (yes if found else no)]
        if len(matches) != 1:
            raise AdapterFailure("ambiguous_presence_options", "No unique Yes/No option for the model result")
        return matches[0]

    @staticmethod
    def _detection_box(result: Any, height: int, width: int) -> list[int]:
        import numpy as np

        try:
            original_size = tuple(result.original_size)
            boxes = np.asarray(result.boxes, dtype=np.float64)
            scores = np.asarray(result.scores, dtype=np.float64)
            count = result.detection_count
        except (AttributeError, TypeError, ValueError):
            raise AdapterFailure("invalid_detection_result", "Invalid detector output") from None
        if original_size != (height, width):
            raise AdapterFailure("image_dimensions_mismatch", "Request dimensions differ from the original image")
        if (boxes.ndim != 2 or boxes.shape[1] != 4
                or scores.shape != (len(boxes),) or count != len(boxes)
                or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
            raise AdapterFailure("invalid_detection_result", "Invalid detector boxes or scores")
        if not len(boxes):
            raise AdapterFailure("empty_detections", "Detection did not find the target")
        index = getattr(result, "grounding_index", None)
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not 0 <= int(index) < len(boxes)):
            raise AdapterFailure("invalid_grounding_match", "Missing valid description-to-box assignment")
        # Selection belongs to the native phrase-map matcher, not detector confidence.
        box = np.clip(boxes[int(index)], 0, [width, height, width, height])
        if box[2] <= box[0] or box[3] <= box[1]:
            raise AdapterFailure("empty_detection_box", "Selected detection has no area in the original image")
        return [int(np.floor(box[0])), int(np.floor(box[1])),
                int(np.ceil(box[2])), int(np.ceil(box[3]))]

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> str | list[int]:
        started = time.monotonic()
        plan = None
        execution_metadata: dict[str, Any] = {}
        _capture_replay_request(request)
        try:
            if len(image_paths) != 1:
                raise AdapterFailure("unsupported_image_count", "Single-image executors require one image")
            plan = parse_question(request)
            answer = self._predict_plan(request, image_paths[0], plan, execution_metadata)
        except TaskInputError as error:
            failure = AdapterFailure(error.code, "Native task input cannot be processed", task=plan.task if plan else None)
            _prediction_diagnostic(request, len(image_paths), plan, started, failure, execution_metadata)
            raise failure from None
        except Exception as error:
            _prediction_diagnostic(request, len(image_paths), plan, started, error, execution_metadata)
            raise
        _prediction_diagnostic(request, len(image_paths), plan, started, execution_metadata=execution_metadata)
        return answer

    def _predict_plan(self, request: dict[str, Any], image: Path,
                      plan: QuestionPlan, execution_metadata: dict[str, Any] | None = None) -> str | list[int]:
        if plan.task not in {"classification", "vqa", "presence", "counting", "visual_grounding"}:
            raise AdapterFailure("unsupported_task", "No registered visual executor")
        if plan.task == "vqa":
            return self._vqa(request, image, self._model_text(plan.original_question), execution_metadata)
        if plan.task == "classification":
            return self._scene(request, image, execution_metadata)
        model_text = self._model_text(plan.original_question)
        if plan.task == "counting":
            if execution_metadata is not None:
                execution_metadata["executor"] = "counting"
            result = self.counting.predict(image, model_text)
            return self._number(request, result.count)
        if plan.task == "presence":
            if execution_metadata is not None:
                execution_metadata["executor"] = "detection"
            result = self.detection.predict(image, model_text, require_grounding=False)
            if execution_metadata is not None:
                execution_metadata["detection_inference"] = getattr(result, "diagnostics", None)
            return self._presence(request, result.detection_count > 0)
        height, width = request["image_height"], request["image_width"]
        if self.grounding_backend == "detection":
            if execution_metadata is not None:
                execution_metadata["executor"] = "detection"
            result = (self.grounding_executor.predict(image, model_text)
                      if self.grounding_executor is not None else self.detection.predict(image, model_text))
            if execution_metadata is not None:
                execution_metadata["detection_inference"] = getattr(result, "diagnostics", None)
            return self._detection_box(result, height, width)
        if execution_metadata is not None:
            execution_metadata["executor"] = "refseg"
        result = self.refseg.predict(image, model_text)
        if result.mask.shape != (height, width):
            raise AdapterFailure("image_dimensions_mismatch", "Request dimensions differ from the original image")
        ys, xs = result.mask.nonzero()
        if len(xs) == 0:
            raise AdapterFailure("empty_refseg_mask", "RefSeg did not find the target")
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
