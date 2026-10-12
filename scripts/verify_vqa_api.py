"""Verify trained Q-Former VQA through the local API on existing public examples.

The fixed-task group checks VQA contracts separately from the unmodified Qwen
router group. Both use the same real predictor. No production key, official
material, public endpoint, training data or old-model expected answer is used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.error
import urllib.request

CODE_DIR = Path(__file__).resolve().parents[1]
API_DIR = CODE_DIR / "participant_api_starter-main"
QFORMER_ARCHITECTURE = "hfsa_frozen_qformer_vqa_v2"
SUPPORTED_HEADS = {"category", "color", "shape", "existence", "scene"}
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(API_DIR))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def example_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Public example path is missing or outside its bundle")
    return path


def load_public_examples(root: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (root / "examples/demo_inputs.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Public example IDs must be nonempty and unique")
    for row in rows:
        example_path(root, row["image"])
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Public example question is missing")
        if row.get("mode") not in {"short_text", "single_choice", "enum"}:
            raise ValueError("Public example answer mode is unsupported")
    return rows


def predictor_options(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("choices", "allowed_answers") if key in row}


def protocol_request(row: dict[str, Any], reference: dict[str, Any], index: int) -> dict[str, Any]:
    mode = row["mode"]
    constraint: dict[str, Any] = {"type": mode}
    extra = {}
    if mode == "single_choice":
        constraint["values"] = list(row["choices"])
        extra["choices"] = row["choices"]
    elif mode == "enum":
        constraint["values"] = row["allowed_answers"]
    elif mode != "short_text":
        raise ValueError("Unsupported public request mode")
    return {
        "protocol_version": "2.0", "request_id": f"public-qformer:{index}:1",
        "item_id": f"PUBLIC_QFORMER_{index}", "question": row["question"],
        "images": [reference], "answer_type": "short_text",
        "response_constraint": constraint, **extra,
    }


def expected_http(request: dict[str, Any], result: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if result["status"] != "ok":
        return 500, {"error": "inference_failed"}
    return 200, {
        "protocol_version": "2.0", "request_id": request["request_id"],
        "item_id": request["item_id"], "answer": result["answer"],
    }


@contextmanager
def disable_verification_capture():
    # This process must not write public test traffic into a production replay
    # directory inherited from a shell. It never reads MODEL_API_KEY.
    key = "HFSA_API_REPLAY_DIR"
    prior = os.environ.get(key)
    os.environ[key] = ""
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


class FixedVQAModel:
    """Exercise the actual VQA adapter with an explicit task; no router patch."""

    def __init__(self, adapter: Any):
        self.adapter = adapter

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> Any:
        from hfsa_adapter import AdapterFailure, QuestionPlan

        if len(image_paths) != 1:
            raise AdapterFailure("unsupported_image_count", "VQA requires one image")
        return self.adapter._predict_plan(
            request, image_paths[0], QuestionPlan("vqa", request["question"]))


def sealed_public_package(root: Path, rows: list[dict[str, Any]], package: Path) -> dict[str, dict[str, Any]]:
    from PIL import Image

    (package / "assets").mkdir()
    assets, references = [], {}
    for row in rows:
        relative = row["image"]
        if relative in references:
            continue
        image = example_path(root, relative)
        payload = image.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        with Image.open(image) as decoded:
            mime_type = Image.MIME[decoded.format]
        reference = {"asset_id": digest[:32], "sha256": digest, "mime_type": mime_type}
        (package / "assets" / reference["asset_id"]).write_bytes(payload)
        # AssetStore requires one manifest entry per digest, even if two public
        # paths refer to identical JPEG bytes.
        if not any(entry["asset_id"] == reference["asset_id"] for entry in assets):
            assets.append(dict(reference, size=len(payload)))
        references[relative] = reference
    (package / "manifest.json").write_text(json.dumps({
        "format": "rsu-sealed-images-v1", "dataset_id": "public-qformer-api-verification",
        "assets": assets,
    }), encoding="utf-8")
    return references


def verify_adapter(adapter: Any, bundle_dir: Path, examples_dir: Path, *, live_qwen: bool = True) -> dict[str, Any]:
    """Reuse an already loaded adapter; never construct another visual model."""
    from model_adapter import RealModel
    from hfsa_adapter import validate_api_vqa_bundle
    from ollama_router import RouterError, route_question
    from server import APIServer, AssetStore

    started = time.perf_counter()
    bundle_dir, examples_dir = Path(bundle_dir).resolve(), Path(examples_dir).resolve()
    predictor = adapter.vqa
    if predictor is None or predictor.manifest["architecture_id"] != QFORMER_ARCHITECTURE:
        raise ValueError("API verification requires the trained Q-Former v2 predictor")
    training_receipt = validate_api_vqa_bundle(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    checkpoint = bundle_dir / saved["checkpoint"]["path"]
    if (saved["architecture_id"] != QFORMER_ARCHITECTURE
            or saved["checkpoint"]["sha256"] != predictor.manifest["checkpoint"]["sha256"]
            or file_sha256(checkpoint) != saved["checkpoint"]["sha256"]):
        raise ValueError("Loaded API predictor does not match the supplied Q-Former bundle")
    if set(predictor.labels["answers"]) != SUPPORTED_HEADS:
        raise ValueError("Q-Former bundle must contain the five supported answer heads")
    identities = (id(predictor), id(predictor.head), id(predictor.visual), id(predictor.text))
    integrity_paths = [manifest_path, checkpoint, Path(__file__),
                       bundle_dir / "training_result.json", bundle_dir / "config/train.json",
                       bundle_dir / "evaluation/metrics.json",
                       API_DIR / "hfsa_adapter.py", API_DIR / "model_adapter.py", API_DIR / "server.py",
                       API_DIR / "ollama_router.py", CODE_DIR / "tasks/vqa/inference.py",
                       examples_dir / "examples/demo_inputs.jsonl"]
    before_hashes = {str(path): file_sha256(path) for path in integrity_paths}
    rows = load_public_examples(examples_dir)
    before_hashes.update({str(example_path(examples_dir, row["image"])):
                         file_sha256(example_path(examples_dir, row["image"])) for row in rows})
    # Reuse the prior public color input but score its enum contract with the
    # current predictor. These options are interface checks, not gold labels.
    color = next(row for row in rows if row["mode"] == "single_choice")
    enum = {key: value for key, value in color.items() if key != "choices"}
    enum.update(sample_id=color["sample_id"] + ":current_api_enum", mode="enum",
                allowed_answers=["Red", "Black"])
    rows.append(enum)
    report: dict[str, Any] = {
        "architecture_id": QFORMER_ARCHITECTURE,
        "checkpoint_sha256": saved["checkpoint"]["sha256"], "device": str(predictor.device),
        "full_training_receipt": training_receipt,
        "scope": "Public interface parity, not benchmark accuracy; fixed-task and live Qwen are separate groups",
        "examples_source": "Existing bundled public VRSBench validation examples; no gold answer file read",
        "enum_protocol": "Constructed Red/Black enum on the bundled public vehicle-color question",
        "other_visual_executors_loaded_by_verifier": False,
        "fixed_task_http": [], "live_qwen_http": [], "live_qwen_skipped": [],
        "errors": [], "checks": {},
    }
    report["checks"]["five_trained_answer_heads"] = set(predictor.labels["answers"]) == SUPPORTED_HEADS
    report["checks"]["head_and_frozen_encoders_eval"] = not (
        predictor.head.training or predictor.visual.training or predictor.text.encoder.training)
    report["checks"]["encoder_parameters_frozen"] = not (
        any(parameter.requires_grad for parameter in predictor.visual.parameters())
        or any(parameter.requires_grad for parameter in predictor.text.parameters()))
    direct = {}
    for row in rows:
        direct[row["sample_id"]] = predictor.predict(
            example_path(examples_dir, row["image"]), row["question"], **predictor_options(row))
    fixed_model = FixedVQAModel(adapter)
    # Reuse RealModel.predict itself while avoiding its full factory and any
    # second encoder/detector/classifier allocation.
    live_model = RealModel.__new__(RealModel)
    live_model.adapter = adapter
    with disable_verification_capture(), tempfile.TemporaryDirectory(prefix="hfsa-qformer-http-") as directory:
        package = Path(directory)
        references = sealed_public_package(examples_dir, rows, package)
        token = secrets.token_urlsafe(32)
        server = APIServer(("127.0.0.1", 0), AssetStore(package), fixed_model, token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def call(path: str, request=None, *, authorized=True):
            headers = {"Content-Type": "application/json"}
            if authorized:
                headers["Authorization"] = "Bearer " + token
            data = None if request is None else json.dumps(request, ensure_ascii=False).encode("utf-8")
            url = f"http://127.0.0.1:{server.server_port}" + path
            try:
                with opener.open(urllib.request.Request(url, data=data, headers=headers), timeout=120) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

        try:
            report["checks"]["unauthorized_health_401"] = call("/healthz", authorized=False)[0] == 401
            report["checks"]["health_ready_200"] = call("/healthz") == (200, {
                "status": "ready", "protocol_version": "2.0", "dataset_id": "public-qformer-api-verification"})
            fixed_types, fixed_formats, live_formats = set(), set(), set()
            fixed_failure_count = 0
            for index, row in enumerate(rows):
                request = protocol_request(row, references[row["image"]], index)
                result = direct[row["sample_id"]]
                expected = expected_http(request, result)
                before = time.perf_counter()
                actual = call("/v1/predict", request)
                record = dict(sample_id=row["sample_id"], mode=row["mode"],
                              direct_result=result, status=actual[0], response=actual[1],
                              seconds=time.perf_counter() - before, matches_direct=(actual == expected))
                report["fixed_task_http"].append(record)
                if actual != expected:
                    report["errors"].append({"sample_id": row["sample_id"], "group": "fixed_task_http", "code": "direct_http_mismatch"})
                if result["status"] == "ok":
                    fixed_types.add(result["question_type"])
                    if actual[0] == 200:
                        fixed_formats.add(row["mode"])
                else:
                    fixed_failure_count += 1
            report["checks"]["five_heads_real_outputs"] = fixed_types == SUPPORTED_HEADS
            report["checks"]["three_formats_fixed_task_200"] = fixed_formats == {"short_text", "single_choice", "enum"}
            report["checks"]["explicit_unsupported_cases"] = fixed_failure_count > 0
            two_image = protocol_request(rows[0], references[rows[0]["image"]], len(rows))
            two_image["images"] *= 2
            report["checks"]["two_image_rejection_500"] = call("/v1/predict", two_image) == (500, {"error": "inference_failed"})
            wrong_hash = protocol_request(rows[0], references[rows[0]["image"]], len(rows) + 1)
            wrong_hash["images"][0] = dict(wrong_hash["images"][0], sha256="0" * 64)
            report["checks"]["wrong_asset_hash_rejection_422"] = call("/v1/predict", wrong_hash)[0] == 422
            if live_qwen:
                server.model = live_model
                for index, row in enumerate(rows):
                    request = protocol_request(row, references[row["image"]], index)
                    try:
                        task = route_question(request)["task"]
                    except RouterError as error:
                        report["errors"].append({"sample_id": row["sample_id"], "group": "live_qwen", "code": error.code})
                        continue
                    if task != "vqa":
                        report["live_qwen_skipped"].append(dict(sample_id=row["sample_id"], mode=row["mode"], task=task))
                        continue
                    before = time.perf_counter()
                    actual = call("/v1/predict", request)
                    expected = expected_http(request, direct[row["sample_id"]])
                    report["live_qwen_http"].append(dict(
                        sample_id=row["sample_id"], mode=row["mode"], task=task,
                        status=actual[0], response=actual[1], matches_direct=(actual == expected),
                        seconds=time.perf_counter() - before))
                    if actual != expected:
                        report["errors"].append({"sample_id": row["sample_id"], "group": "live_qwen_http", "code": "direct_http_mismatch"})
                    if actual[0] == 200:
                        live_formats.add(row["mode"])
                report["checks"]["three_formats_unpatched_qwen_200"] = live_formats == {"short_text", "single_choice", "enum"}
            report.update(actual_supported_heads=sorted(fixed_types), fixed_formats=sorted(fixed_formats),
                          live_formats=sorted(live_formats), unsupported_case_count=fixed_failure_count,
                          live_qwen_requested=live_qwen)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
    report["checks"]["persistent_predictor_and_encoders"] = identities == (
        id(adapter.vqa), id(predictor.head), id(predictor.visual), id(predictor.text))
    report["checks"]["source_and_bundle_unchanged"] = before_hashes == {
        path: file_sha256(Path(path)) for path in before_hashes}
    report["source_and_bundle_sha256"] = before_hashes
    report["public_image_sha256"] = {relative: reference["sha256"] for relative, reference in references.items()}
    report["seconds"] = time.perf_counter() - started
    report["status"] = "passed" if all(report["checks"].values()) and not report["errors"] else "failed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, default=CODE_DIR / "runs/vqa/stage1")
    parser.add_argument("--models-dir", type=Path, default=CODE_DIR.parent / "HFSA_models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-live-qwen", action="store_true", help="Only fixed-task verification; report records the omitted router group")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    import torch
    from hfsa_adapter import HFSAAdapter, validate_api_vqa_bundle
    from tasks.vqa.inference import VQAPredictor

    torch.set_num_threads(args.threads)
    validate_api_vqa_bundle(args.bundle_dir)
    predictor = VQAPredictor(args.bundle_dir, args.device, {
        "visual_backbone": args.models_dir / "pretrain_model/yolov12m.pt",
        "text_encoder": args.models_dir / "openclip/ViT-L-14.pt",
    })
    adapter = HFSAAdapter(None, None, None, None, vqa=predictor)
    report = verify_adapter(adapter, args.bundle_dir, args.examples_dir, live_qwen=not args.skip_live_qwen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "architecture_id", "checkpoint_sha256", "checks", "errors")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
