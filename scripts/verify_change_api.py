"""Verify one real CPU MCI instance on four preselected public test pairs.

Only image A/B enter the visual model. Public masks/captions are read by the
scorer, never by the model, local Qwen answerer, or task router. The HTTP server
uses a temporary sealed public package, random in-memory credential, and an
ephemeral loopback port. It does not touch the production API or its credential.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
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
DATASET_ID = "public-mci-api-verification"
EXPECTED_IDS = {"test_000001": False, "test_000002": False,
                "test_000004": True, "test_000017": True}
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(API_DIR))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Public material is missing or outside its fixed directory")
    return path


def load_public_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("public_only") is not True
            or manifest.get("official_private_material_used") is not False
            or manifest.get("prediction_based_selection") is not False
            or manifest.get("status") != "acquisition_complete_no_model_inference"
            or manifest.get("dataset", {}).get("id") != "lcybuaa/LEVIR-MCI"):
        raise ValueError("Only the preselected public author test material is allowed")
    rows = manifest["samples"]
    if (len(rows) != 4 or {row["id"]: row["expected_change"] for row in rows} != EXPECTED_IDS
            or any(row.get("split") != "test" for row in rows)):
        raise ValueError("The fixed two-changed/two-unchanged selection must not change")
    receipts = {item["path"]: item for item in manifest["files"]}
    for item in manifest["files"]:
        path = public_path(root, item["path"])
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError("Public source hash/size mismatch")
    for row in rows:
        for key in ("image_a", "image_b", "label", "caption_tokens"):
            if row[key] not in receipts:
                raise ValueError("Public sample lacks a source/hash receipt")
    return manifest


def confusion_metrics(matrix: list[list[int]]) -> dict[str, Any]:
    if (len(matrix) != 3 or any(len(row) != 3 for row in matrix)
            or any(type(value) is not int or value < 0 for row in matrix for value in row)):
        raise ValueError("Expected a nonnegative 3x3 integer confusion matrix")
    total = sum(map(sum, matrix))
    iou = {}
    for index, name in enumerate(("background", "road", "building")):
        union = sum(matrix[index]) + sum(row[index] for row in matrix) - matrix[index][index]
        iou[name] = matrix[index][index] / union if union else None
    valid = [value for value in iou.values() if value is not None]
    return {"confusion_gt_rows_prediction_columns": matrix, "pixels": total,
            "pixel_accuracy": sum(matrix[i][i] for i in range(3)) / total if total else None,
            "iou_by_class": iou, "mean_iou_present_union_classes": sum(valid) / len(valid) if valid else None}


def score_mask(label: Path, predicted: Any) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    with Image.open(label) as image:
        rgb = np.asarray(image.convert("RGB"))
    if (rgb.shape != (256, 256, 3) or not np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            or not np.array_equal(rgb[:, :, 0], rgb[:, :, 2])):
        raise ValueError("Unexpected public GT palette/shape")
    raw = rgb[:, :, 0]
    if not set(np.unique(raw)).issubset({0, 128, 255}):
        raise ValueError("Public GT must preserve the author 0/128/255 palette")
    ground_truth = np.zeros_like(raw)
    ground_truth[raw == 128], ground_truth[raw == 255] = 1, 2
    if predicted.shape != (256, 256) or not set(np.unique(predicted)).issubset({0, 1, 2}):
        raise ValueError("Invalid actual MCI mask shape/classes")
    matrix = np.bincount((3 * ground_truth + predicted).reshape(-1), minlength=9).reshape(3, 3)
    return confusion_metrics(matrix.tolist())


def evidence_record(value: Any) -> dict[str, Any]:
    import numpy as np

    return {"caption": value.caption, "elapsed_seconds": value.elapsed_seconds,
            "original_hw": list(value.original_hw), "mask_hw": list(value.mask.shape),
            "mask_sha256": hashlib.sha256(value.mask.tobytes()).hexdigest(),
            "predicted_pixels": {name: int((value.mask == index).sum())
                                 for index, name in enumerate(("background", "road", "building"))},
            "mask_unique_classes": [int(x) for x in np.unique(value.mask)]}


def yes_no_result(result: Any) -> bool | None:
    if not result.supported:
        return None
    return {"Yes": True, "No": False}.get(result.answer)


def protocol_request(case: dict[str, Any], references: dict[str, Any], index: int) -> dict[str, Any]:
    request = {"protocol_version": "2.0", "request_id": f"public-mci:{index}:1",
               "item_id": f"PUBLIC_MCI_{index}", "question": case["question"],
               "images": [references[path] for path in case["images"]],
               "answer_type": case.get("answer_type", "short_text"),
               "response_constraint": case["constraint"]}
    if case.get("choices") is not None:
        request["choices"] = case["choices"]
    if case["constraint"]["type"] == "bbox":
        request.update(image_width=256, image_height=256, answer_type="bbox")
    return request


def expected_http(request: dict[str, Any], result: Any) -> tuple[int, dict[str, Any]]:
    if not result.supported:
        return 500, {"error": "inference_failed"}
    return 200, {"protocol_version": "2.0", "request_id": request["request_id"],
                 "item_id": request["item_id"], "answer": result.answer}


@contextmanager
def disabled_capture():
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


class FixedChangeModel:
    """Call the real change adapter explicitly, without a router patch."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> str:
        from hfsa_adapter import AdapterFailure

        if len(image_paths) != 2:
            raise AdapterFailure("unsupported_image_count", "Change requires two images")
        return self.adapter._change(request, image_paths[0], image_paths[1], request["question"])


def sealed_package(root: Path, rows: list[dict[str, Any]], package: Path) -> dict[str, Any]:
    (package / "assets").mkdir()
    references, entries = {}, {}
    for row in rows:
        for key in ("image_a", "image_b"):
            relative = row[key]
            payload = public_path(root, relative).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            reference = {"asset_id": digest[:32], "sha256": digest, "mime_type": "image/png"}
            (package / "assets" / reference["asset_id"]).write_bytes(payload)
            entries[reference["asset_id"]] = dict(reference, size=len(payload))
            references[relative] = reference
    (package / "manifest.json").write_text(json.dumps({"format": "rsu-sealed-images-v1",
        "dataset_id": DATASET_ID, "assets": list(entries.values())}), encoding="utf-8")
    return references


def resources() -> dict[str, Any]:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        return {"process_peak_rss_bytes": peak, "cpu_user_seconds": usage.ru_utime,
                "cpu_system_seconds": usage.ru_stime}
    except ImportError:
        return {"process_peak_rss_bytes": None, "cpu_user_seconds": None, "cpu_system_seconds": None}


def save_report(output: Path, report: dict[str, Any]) -> None:
    """Keep this runner's earlier failed/interrupted evidence on a later run."""
    history = []
    if output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("report_type") == "public_mci_api_verification":
            history = prior.get("previous_attempts", [])
            if prior.get("status") in {"failed", "interrupted"}:
                history.append({key: value for key, value in prior.items() if key != "previous_attempts"})
    report["report_type"] = "public_mci_api_verification"
    report["previous_attempts"] = history
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def module_versions(predictor: Any) -> dict[str, Any]:
    return {f"{name}.{key}": (id(tensor), tensor._version, list(tensor.shape))
            for name, module in (("encoder", predictor.encoder), ("fusion", predictor.fusion),
                                 ("decoder", predictor.decoder))
            for key, tensor in list(module.named_parameters()) + list(module.named_buffers())}


def evidence_timing(predictor: Any, start: int) -> dict[str, Any]:
    """Current call measurements, distinct from a cached evidence's old timing."""
    events = getattr(predictor, "verification_evidence_events", [])[start:]
    return {"calls": [dict(event) for event in events],
            "real_visual_evidence_seconds": sum(event["real_visual_seconds"] for event in events),
            "cache_hit_return_seconds": sum(event["evidence_call_seconds"] for event in events if event["cache_hit"]),
            "image_hashing_seconds": sum(event["key_hashing_seconds"] for event in events),
            "scope": "Visual time is actual uncached ChangePredictor.evidence wall time. Cache-return time includes image hashing/lookup; hashing is a nested measurement, not additive. HTTP wall time includes actual local Qwen calls."}


def cases_for(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = {row["id"]: row for row in rows}
    pair = lambda key: [samples[key]["image_a"], samples[key]["image_b"]]
    yes_no = {"type": "enum", "values": ["Yes", "No"]}
    cases = [{"name": "public_binary_" + row["id"], "sample_id": row["id"],
              "images": pair(row["id"]), "question": "Do the earlier and later images show any change? Answer Yes or No.",
              "constraint": yes_no} for row in rows]
    cases += [
        {"name": "caption_short_text", "sample_id": "test_000004", "images": pair("test_000004"),
         "question": "Describe the changes between the earlier image A and the later image B in one short sentence.",
         "constraint": {"type": "short_text", "min_length": 1}},
        {"name": "original_single_choice_key", "sample_id": "test_000017", "images": pair("test_000017"),
         "question": "Compared with the earlier image A, has the later image B changed? Select the original option key.",
         "constraint": {"type": "single_choice", "values": ["changed:key", "same-key"]},
         "choices": {"changed:key": "Yes", "same-key": "No"}},
        {"name": "four_quadrant_region", "sample_id": "test_000004", "images": pair("test_000004"),
         "question": "Comparing the earlier and later images, which quadrant contains the MOST overall road and building change area? Choose one of the four quadrants.",
         "constraint": {"type": "enum", "values": ["upper_left", "upper_right", "lower_left", "lower_right"]},
         "answer_type": "region_label"},
        {"name": "unsupported_exact_object_count", "sample_id": "test_000004", "images": pair("test_000004"),
         "question": "How many individual new buildings have been constructed between the earlier and later image? Return the exact object count.",
         "constraint": {"type": "short_text"}},
        {"name": "unsupported_bbox", "sample_id": "test_000004", "images": pair("test_000004"),
         "question": "Return the bounding box of the newly constructed building between the earlier and later images.",
         "constraint": {"type": "bbox", "coordinate_format": "xyxy", "length": 4}},
        {"name": "unsupported_absent_predicate", "sample_id": "test_000004", "images": pair("test_000004"),
         "question": "Did an airport runway become flooded with water between the earlier and later images? Answer Yes or No.",
         "constraint": yes_no},
        {"name": "unsupported_least_change_region", "sample_id": "test_000004", "images": pair("test_000004"),
         "question": "Comparing the earlier and later images, which quadrant contains the LEAST overall road and building change area? Choose one of the four quadrants.",
         "constraint": {"type": "enum", "values": ["upper_left", "upper_right", "lower_left", "lower_right"]},
         "answer_type": "region_label"},
        {"name": "unsupported_newly_constructed_subset_region", "sample_id": "test_000004", "images": pair("test_000004"),
         "question": "Between the earlier and later images, which quadrant contains the MOST newly constructed buildings, excluding demolished buildings? Choose one of the four quadrants.",
         "constraint": {"type": "enum", "values": ["upper_left", "upper_right", "lower_left", "lower_right"]},
         "answer_type": "region_label"},
    ]
    return cases


def verify_predictor(predictor: Any, bundle_dir: Path, public_dir: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from hfsa_adapter import HFSAAdapter
    from model_adapter import RealModel
    from ollama_router import RouterError, route_question
    from server import APIServer, AssetStore

    started = time.perf_counter()
    public_dir, bundle_dir = public_dir.resolve(), bundle_dir.resolve()
    public = load_public_manifest(public_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["architecture_id"] != "change_agent_mci_v1"
            or manifest["checkpoint"]["sha256"] != predictor.checkpoint_sha256
            or str(predictor.device) != "cpu"):
        raise ValueError("Verification requires the actual supplied MCI bundle on CPU")
    paths = [Path(__file__), bundle_dir / "manifest.json", bundle_dir / manifest["checkpoint"]["path"],
             bundle_dir / manifest["vocabulary"]["path"],
             bundle_dir / manifest["checkpoint_inactive_keys"]["path"], public_dir / "data_manifest.json",
             API_DIR / "hfsa_adapter.py", API_DIR / "model_adapter.py", API_DIR / "server.py",
             API_DIR / "ollama_router.py", CODE_DIR / "tasks/change/inference.py",
             CODE_DIR / "tasks/change/answering.py", *sorted((CODE_DIR / "tasks/change/_mci").rglob("*.py")),
             *(public_path(public_dir, file["path"]) for file in public["files"])]
    before_hashes = {str(path): file_sha256(path) for path in paths}
    if before_hashes[str(bundle_dir / manifest["checkpoint"]["path"])] != predictor.checkpoint_sha256:
        raise ValueError("Actual checkpoint SHA differs from loaded MCI")
    adapter = HFSAAdapter(None, None, None, None, change=predictor)
    modules = (predictor.encoder, predictor.fusion, predictor.decoder)
    identities = (id(predictor), *(id(module) for module in modules))
    versions = module_versions(predictor)
    report = {"architecture_id": manifest["architecture_id"], "checkpoint_sha256": predictor.checkpoint_sha256,
              "device": "cpu", "parameter_count": predictor.parameter_count, "strict_loading": predictor.loading,
              "inactive_checkpoint_keys_sha256": before_hashes[str(bundle_dir / manifest["checkpoint_inactive_keys"]["path"])],
              "public_manifest_sha256": before_hashes[str(public_dir / "data_manifest.json")],
              "selection_rule": public["selection_rule"], "selected_ids": list(EXPECTED_IDS),
              "scope": "Four preselected public test pairs; pixel/binary diagnostics are separate from HTTP parity. No full-set or official score.",
              "ground_truth_role": "Scoring only; only image paths and synthetic public questions/options enter inference.",
              "mask_classes": ["background", "road", "building"], "mask_temporal_direction": False,
              "samples": [], "controls": {}, "fixed_task_http": [], "live_qwen_http": [],
              "uncached_live_qwen_http": None,
              "checks": {}, "errors": [], "production_api_started_or_restarted": False,
              "production_credential_read": False, "private_material_used": False}
    report["checks"]["strict_three_part_loading"] = (
        set(predictor.loading) == {"encoder_dict", "encoder_trans_dict", "decoder_dict"}
        and all(value.get("missing") == 0 and value.get("unexpected") == 0
                and value.get("strict_active_state") is True
                and value.get("matched_tensors", 0) > 0 for value in predictor.loading.values()))
    report["loading_summary"] = {
        "matched_tensors": sum(value["matched_tensors"] for value in predictor.loading.values()),
        "ignored_inactive_tensors": sum(value["ignored_inactive_tensors"] for value in predictor.loading.values()),
        "checkpoint_tensors": sum(value["checkpoint_tensors"] for value in predictor.loading.values()),
        "strict_active_state": all(value["strict_active_state"] is True for value in predictor.loading.values()),
        "scope": "All 1194 active-state tensors load strictly; only the exact source-audited 942 inactive author leftovers are ignored."}
    report["checks"]["audited_active_inactive_loading_counts"] = (
        report["loading_summary"]["matched_tensors"] == 1194
        and report["loading_summary"]["ignored_inactive_tensors"] == 942
        and report["loading_summary"]["checkpoint_tensors"] == 2136
        and predictor.parameter_count == 130502679
        and report["inactive_checkpoint_keys_sha256"] == "a36086972665b28d5a67a36d8d19f3e972afccb6d741dff346539f5abbbec50c")
    report["checks"]["all_modules_frozen_eval_cpu"] = all(
        not module.training and all(not p.requires_grad and p.device.type == "cpu" for p in module.parameters())
        for module in modules)
    evidence = {}
    total_matrix = [[0] * 3 for _ in range(3)]
    for row in public["samples"]:
        # No GT/reference values are passed into this real model call.
        value = predictor.evidence(public_path(public_dir, row["image_a"]), public_path(public_dir, row["image_b"]))
        evidence[row["id"]] = value
        score = score_mask(public_path(public_dir, row["label"]), value.mask)
        for i in range(3):
            for j in range(3):
                total_matrix[i][j] += score["confusion_gt_rows_prediction_columns"][i][j]
        report["samples"].append(dict(sample_id=row["id"], expected_change=row["expected_change"],
            actual_evidence=evidence_record(value), segmentation=score,
            caption_exact_reference_match=value.caption.strip(" .").lower() in
                {text.strip(" .").lower() for text in row["captions"]}))
        print(json.dumps({"stage": "public_evidence_scored", "sample_id": row["id"],
                          "seconds": value.elapsed_seconds}), flush=True)
    report["aggregate_segmentation"] = confusion_metrics(total_matrix)
    case_rows = cases_for(public["samples"])
    direct = {}
    for case in case_rows:
        print(json.dumps({"stage": "direct_change_question", "case": case["name"]}), flush=True)
        direct[case["name"]] = predictor.predict(*(public_path(public_dir, path) for path in case["images"]),
            case["question"], response_constraint=case["constraint"], choices=case.get("choices"))
    correct = covered = 0
    for row in report["samples"]:
        result = direct["public_binary_" + row["sample_id"]]
        prediction = yes_no_result(result)
        row["binary_question_result"] = asdict(result)
        row["binary_prediction"] = prediction
        row["binary_correct"] = prediction is not None and prediction == row["expected_change"]
        covered += prediction is not None
        correct += row["binary_correct"]
    report["binary_diagnostic"] = {"n": 4, "covered": covered, "correct": correct,
                                    "accuracy_all_four_denominator": correct / 4}
    unchanged = [row for row in report["samples"] if row["expected_change"] is False]
    report["binary_diagnostic"]["no_change_samples"] = {
        "n": len(unchanged), "correct": sum(row["binary_correct"] for row in unchanged)}
    report["checks"]["two_public_no_change_binary_correct"] = (
        len(unchanged) == 2 and all(row["binary_correct"] for row in unchanged))
    report["checks"]["all_four_public_binary_correct"] = covered == 4 and correct == 4
    for row in report["samples"]:
        if not row["binary_correct"]:
            report["errors"].append({"group": "binary_diagnostic", "case": row["sample_id"],
                                      "code": "public_binary_wrong_or_refused"})
    report["capability_refusals"] = {case["name"]: asdict(direct[case["name"]])
                                    for case in case_rows if case["name"].startswith("unsupported_")}
    report["checks"]["unsupported_capability_questions_refused"] = all(
        not result["supported"] for result in report["capability_refusals"].values())
    for name, result in report["capability_refusals"].items():
        if result["supported"]:
            report["errors"].append({"group": "capability_refusals", "case": name,
                                      "code": "unsupported_capability_was_answered"})
    foreground = evidence["test_000004"].mask != 0
    region_pixels = {"upper_left": int(foreground[:128, :128].sum()),
                     "upper_right": int(foreground[:128, 128:].sum()),
                     "lower_left": int(foreground[128:, :128].sum()),
                     "lower_right": int(foreground[128:, 128:].sum())}
    maximum = max(region_pixels.values())
    winners = [name for name, pixels in region_pixels.items() if pixels == maximum]
    mask_winner = winners[0] if maximum > 0 and len(winners) == 1 else None
    result = direct["four_quadrant_region"]
    region_matches = not result.supported or (mask_winner is not None and result.answer == mask_winner)
    report["region_mask_diagnostic"] = {"predicted_changed_pixels_by_quadrant": region_pixels,
        "unique_positive_maximum_from_actual_mask": mask_winner,
        "actual_question_result": asdict(result),
        "supported_answer_matches_actual_mask": region_matches if result.supported else None,
        "scope": "Actual model mask consistency only; a refusal remains a refusal, and no GT enters this comparison."}
    report["checks"]["supported_maximum_region_matches_actual_mask"] = region_matches
    if not region_matches:
        report["errors"].append({"group": "region_mask_diagnostic", "case": "four_quadrant_region",
                                  "code": "supported_region_not_actual_mask_maximum"})
    # Perturbations are model sensitivity/shape checks, not reverse-order GT.
    print(json.dumps({"stage": "real_image_pair_controls"}), flush=True)
    changed = next(row for row in public["samples"] if row["id"] == "test_000004")
    a, b = (public_path(public_dir, changed[key]) for key in ("image_a", "image_b"))
    same = predictor.evidence(a, a)
    reverse = predictor.evidence(b, a)
    same_answer = predictor.predict(a, a, "Do the earlier and later images show any change? Answer Yes or No.",
                                    response_constraint={"type": "enum", "values": ["Yes", "No"]})
    report["controls"]["same_image"] = dict(actual_evidence=evidence_record(same),
        binary_question_result=asdict(same_answer), expected_change_by_construction=False,
        binary_correct=yes_no_result(same_answer) is False)
    report["checks"]["same_image_binary_correct_no"] = yes_no_result(same_answer) is False
    if not report["checks"]["same_image_binary_correct_no"]:
        report["errors"].append({"group": "controls", "case": "same_image",
                                  "code": "same_image_binary_wrong_or_refused"})
    report["controls"]["swapped_A_B"] = dict(actual_evidence=evidence_record(reverse),
        caption_changed_from_forward=reverse.caption != evidence[changed["id"]].caption,
        mask_agreement_with_forward=float(np.mean(reverse.mask == evidence[changed["id"]].mask)),
        scored_against_forward_gt=False, note="Reversed caption direction is observed, not assumed; mask has no added/removed encoding.")
    with tempfile.TemporaryDirectory(prefix="hfsa-mci-resize-") as directory:
        scaled = []
        for name, path in (("A", a), ("B", b)):
            output = Path(directory) / (name + ".png")
            with Image.open(path) as image:
                image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR).save(output)
            scaled.append(output)
        value = predictor.evidence(*scaled)
        report["controls"]["resize_512"] = dict(actual_evidence=evidence_record(value),
            preparation="PIL bilinear 256->512; real Predictor OpenCV linear 512->256",
            caption_matches_native=value.caption == evidence[changed["id"]].caption,
            mask_agreement_with_native=float(np.mean(value.mask == evidence[changed["id"]].mask)))
    report["checks"]["resize_512_preserves_original_hw_and_output_shape"] = (
        value.original_hw == (512, 512) and value.mask.shape == (256, 256))
    live_model = RealModel.__new__(RealModel)
    live_model.adapter = adapter
    with disabled_capture(), tempfile.TemporaryDirectory(prefix="hfsa-mci-http-") as directory:
        package = Path(directory)
        references = sealed_package(public_dir, public["samples"], package)
        token = secrets.token_urlsafe(32)
        server = APIServer(("127.0.0.1", 0), AssetStore(package), FixedChangeModel(adapter), token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def call(path: str, request=None, *, authorized=True, timeout_seconds=100):
            headers = {"Content-Type": "application/json"}
            if authorized:
                headers["Authorization"] = "Bearer " + token
            data = None if request is None else json.dumps(request, ensure_ascii=False).encode("utf-8")
            try:
                with opener.open(urllib.request.Request(f"http://127.0.0.1:{server.server_port}" + path,
                                                       data=data, headers=headers), timeout=timeout_seconds) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

        try:
            report["checks"]["unauthorized_health_401"] = call("/healthz", authorized=False) == (401, {"error": "unauthorized"})
            report["checks"]["health_ready_200"] = call("/healthz") == (200, {
                "status": "ready", "protocol_version": "2.0", "dataset_id": DATASET_ID})
            fixed_formats = set()
            for index, case in enumerate(case_rows):
                print(json.dumps({"stage": "fixed_task_real_http", "case": case["name"]}), flush=True)
                request = protocol_request(case, references, index)
                before = time.perf_counter()
                event_start = len(predictor.verification_evidence_events)
                actual = call("/v1/predict", request)
                expected = expected_http(request, direct[case["name"]])
                report["fixed_task_http"].append(dict(name=case["name"], sample_id=case["sample_id"],
                    question=case["question"], constraint=case["constraint"], choices=case.get("choices"),
                    direct_result=asdict(direct[case["name"]]), status=actual[0], response=actual[1],
                    matches_direct=actual == expected, seconds=time.perf_counter() - before,
                    evidence_timing=evidence_timing(predictor, event_start)))
                if actual != expected:
                    report["errors"].append({"group": "fixed_task_http", "case": case["name"], "code": "direct_http_mismatch"})
                if actual[0] == 200:
                    fixed_formats.add(case["constraint"]["type"])
            report["checks"]["three_real_weight_formats_http_200"] = fixed_formats >= {"short_text", "enum", "single_choice"}
            report["checks"]["region_real_weight_http_observed"] = any(
                row["name"] == "four_quadrant_region" and row["matches_direct"] for row in report["fixed_task_http"])
            one = protocol_request(case_rows[0], references, len(case_rows))
            one["images"] = one["images"][:1]
            report["checks"]["one_image_rejected_500"] = call("/v1/predict", one) == (500, {"error": "inference_failed"})
            bad = protocol_request(case_rows[0], references, len(case_rows) + 1)
            bad["images"][0] = dict(bad["images"][0], sha256="0" * 64)
            report["checks"]["wrong_image_hash_rejected_422"] = call("/v1/predict", bad)[0] == 422
            report["checks"]["unauthorized_predict_401"] = call("/v1/predict", one, authorized=False)[0] == 401
            server.model = live_model
            live_names = {"public_binary_test_000001", "caption_short_text", "original_single_choice_key", "four_quadrant_region"}
            for index, case in enumerate(case_rows):
                if case["name"] not in live_names:
                    continue
                print(json.dumps({"stage": "live_unpatched_qwen_http", "case": case["name"]}), flush=True)
                request = protocol_request(case, references, len(case_rows) + index + 2)
                try:
                    task, routing_error = route_question(request)["task"], None
                except RouterError as error:
                    task, routing_error = None, error.code
                before = time.perf_counter()
                event_start = len(predictor.verification_evidence_events)
                actual = call("/v1/predict", request)
                expected = expected_http(request, direct[case["name"]])
                report["live_qwen_http"].append(dict(name=case["name"], task=task, routing_error=routing_error,
                    status=actual[0], response=actual[1], matches_direct=actual == expected,
                    seconds=time.perf_counter() - before,
                    evidence_timing=evidence_timing(predictor, event_start)))
                if task != "change" or actual != expected:
                    report["errors"].append({"group": "live_qwen_http", "case": case["name"],
                                             "code": routing_error or "routing_or_direct_http_mismatch"})
            report["checks"]["four_live_unpatched_qwen_http_cases"] = len(report["live_qwen_http"]) >= 4
            # A separate full HTTP request must execute real visual inference.
            # Eviction changes only this runner's bytes-keyed evidence cache.
            case = next(row for row in case_rows if row["name"] == "caption_short_text")
            evicted = predictor.evict_verification_pair(
                *(public_path(public_dir, path) for path in case["images"]))
            request = protocol_request(case, references, len(case_rows) * 3)
            print(json.dumps({"stage": "uncached_live_qwen_full_model_http", "case": case["name"]}), flush=True)
            try:
                task, routing_error = route_question(request)["task"], None
            except RouterError as error:
                task, routing_error = None, error.code
            event_start = len(predictor.verification_evidence_events)
            before = time.perf_counter()
            actual = call("/v1/predict", request, timeout_seconds=240)
            elapsed = time.perf_counter() - before
            timing = evidence_timing(predictor, event_start)
            expected = expected_http(request, direct[case["name"]])
            report["uncached_live_qwen_http"] = dict(name=case["name"], task=task,
                routing_error=routing_error, question=case["question"], cache_entry_evicted=evicted,
                status=actual[0], response=actual[1], matches_direct=actual == expected,
                seconds=elapsed, evidence_timing=timing,
                http_timeout_seconds=240,
                note="Actual APIServer + RealModel.predict + unpatched Qwen task router + uncached real MCI + actual Qwen answerer.")
            report["checks"]["one_uncached_complete_live_qwen_model_http"] = (
                task == "change" and actual == expected and actual[0] == 200
                and len(timing["calls"]) == 1 and timing["calls"][0]["cache_hit"] is False
                and timing["real_visual_evidence_seconds"] > 0)
            if not report["checks"]["one_uncached_complete_live_qwen_model_http"]:
                report["errors"].append({"group": "uncached_live_qwen_http", "case": case["name"],
                    "code": routing_error or "uncached_full_model_http_not_verified"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        report["checks"]["temporary_http_server_closed"] = not thread.is_alive()
    report["checks"]["same_predictor_and_modules_reused"] = identities == (
        id(adapter.change), id(predictor.encoder), id(predictor.fusion), id(predictor.decoder))
    report["checks"]["parameter_buffer_versions_unchanged"] = versions == module_versions(predictor)
    report["checks"]["source_bundle_public_material_unchanged"] = before_hashes == {
        path: file_sha256(Path(path)) for path in before_hashes}
    report["source_bundle_public_sha256"] = before_hashes
    report["http_request_counts"] = {"fixed_task_prediction_cases": len(report["fixed_task_http"]),
        "live_qwen_prediction_cases": len(report["live_qwen_http"]),
        "live_qwen_uncached_prediction_cases": int(report["uncached_live_qwen_http"] is not None),
        "additional_boundary_predict_requests": 3, "health_requests": 2,
        "total_actual_http_requests": len(report["fixed_task_http"]) + len(report["live_qwen_http"])
            + int(report["uncached_live_qwen_http"] is not None) + 5}
    report["resources_after_verification"] = resources()
    report["resource_scope"] = "Verifier Python process only; local Ollama runs in a separate process."
    report["caption_exact_match_scope"] = "Lexical reference overlap diagnostic only; not semantic caption accuracy."
    report["verification_seconds"] = time.perf_counter() - started
    report["status"] = "passed" if all(report["checks"].values()) and not report["errors"] else "failed"
    return report


def self_check() -> None:
    from types import SimpleNamespace
    from hfsa_adapter import AdapterFailure

    perfect = confusion_metrics([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
    assert perfect["pixel_accuracy"] == 1 and perfect["iou_by_class"] == {"background": 1, "road": 1, "building": 1}
    assert confusion_metrics([[2, 0, 0], [0, 0, 0], [0, 0, 0]])["iou_by_class"]["road"] is None
    seen = []
    adapter = SimpleNamespace(_change=lambda request, a, b, question: seen.append((a, b, question)) or "key:X")
    request = {"question": "  original temporal question\n", "request_id": "r", "item_id": "i"}
    assert FixedChangeModel(adapter).predict(request, [Path("A"), Path("B")]) == "key:X"
    assert seen == [(Path("A"), Path("B"), request["question"])]
    try:
        FixedChangeModel(adapter).predict(request, [Path("A")])
    except AdapterFailure:
        pass
    else:
        raise AssertionError("One-image contract must reject")
    assert expected_http(request, SimpleNamespace(supported=False)) == (500, {"error": "inference_failed"})
    assert expected_http(request, SimpleNamespace(supported=True, answer="key:X"))[1]["answer"] == "key:X"
    timing = evidence_timing(SimpleNamespace(verification_evidence_events=[
        {"cache_hit": True, "key_hashing_seconds": 0.001,
         "evidence_call_seconds": 0.002, "real_visual_seconds": 0.0},
        {"cache_hit": False, "key_hashing_seconds": 0.003,
         "evidence_call_seconds": 3.004, "real_visual_seconds": 3.0}]), 0)
    assert timing["real_visual_evidence_seconds"] == 3.0
    assert timing["cache_hit_return_seconds"] == 0.002
    assert timing["image_hashing_seconds"] == 0.004
    with tempfile.TemporaryDirectory() as directory:
        report_path = Path(directory) / "synthetic-report.json"
        save_report(report_path, {"status": "interrupted", "stage": "synthetic-one"})
        save_report(report_path, {"status": "failed", "stage": "synthetic-two"})
        save_report(report_path, {"status": "passed", "stage": "synthetic-three"})
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        assert [row["stage"] for row in saved["previous_attempts"]] == ["synthetic-one", "synthetic-two"]
        root = Path(directory) / "public"
        root.mkdir()
        (Path(directory) / "outside").write_bytes(b"synthetic")
        try:
            public_path(root, "../outside")
        except ValueError:
            pass
        else:
            raise AssertionError("Public paths must not escape")
    print(json.dumps({"status": "self_check_passed", "scope": "Synthetic helper boundaries only; no model was loaded"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=CODE_DIR / "runs/change/mci")
    parser.add_argument("--public-dir", type=Path, default=CODE_DIR / "runs/change/public_validation")
    parser.add_argument("--output", type=Path, default=CODE_DIR / "runs/change/mci/reports/public_api_verification.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-check", action="store_true", help="Check synthetic helper boundaries without torch/model loading")
    args = parser.parse_args()
    if args.threads != 4:
        parser.error("This controlled CPU verification requires --threads 4")
    if args.self_check:
        self_check()
        return
    started = time.perf_counter()
    before_resources = resources()
    print(json.dumps({"stage": "importing_torch_and_change_predictor"}), flush=True)
    import torch
    from tasks.change.inference import ChangePredictor

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    # An evidence-only cache in this runner's subclass. Production predict and
    # its actual Qwen answerer remain inherited and unmodified. Cache keys use
    # image bytes only, so sealed HTTP copies reuse the same real evidence.
    class CachedVerificationPredictor(ChangePredictor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.verification_cache = {}
            self.verification_cache_hits = 0
            self.verification_evidence_events = []

        def evidence(self, before, after):
            started = time.perf_counter()
            key = (file_sha256(Path(before)), file_sha256(Path(after)))
            hashing_seconds = time.perf_counter() - started
            if key in self.verification_cache:
                self.verification_cache_hits += 1
                value = self.verification_cache[key]
                self.verification_evidence_events.append({"cache_hit": True,
                    "key_hashing_seconds": hashing_seconds,
                    "evidence_call_seconds": time.perf_counter() - started,
                    "real_visual_seconds": 0.0})
                return value
            value = super().evidence(before, after)
            self.verification_cache[key] = value
            self.verification_evidence_events.append({"cache_hit": False,
                "key_hashing_seconds": hashing_seconds,
                "evidence_call_seconds": time.perf_counter() - started,
                "real_visual_seconds": value.elapsed_seconds})
            return value

        def evict_verification_pair(self, before, after):
            key = (file_sha256(Path(before)), file_sha256(Path(after)))
            return self.verification_cache.pop(key, None) is not None

    imports_seconds = time.perf_counter() - started
    print(json.dumps({"stage": "strict_loading_real_MCI", "device": "cpu", "threads": 4}), flush=True)
    loading_started = time.perf_counter()
    try:
        predictor = CachedVerificationPredictor(bundle_dir=args.bundle_dir, device="cpu")
        loading_seconds = time.perf_counter() - loading_started
        print(json.dumps({"stage": "real_MCI_loaded", "parameter_count": predictor.parameter_count,
                          "seconds": loading_seconds}), flush=True)
        report = verify_predictor(predictor, args.bundle_dir, args.public_dir)
        report.update(imports_seconds=imports_seconds, loading_seconds=loading_seconds,
                      resources_before_import=before_resources, resources_after_verification=resources(),
                      torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                      cached_real_evidence_pairs=len(predictor.verification_cache),
                      cached_real_evidence_hits=predictor.verification_cache_hits,
                      real_visual_evidence_calls=sum(not event["cache_hit"]
                          for event in predictor.verification_evidence_events),
                      all_evidence_call_timing=evidence_timing(predictor, 0),
                      evidence_cache_scope="Runner only; inherited real production predict and answerer; no GT in cache keys or values",
                      total_seconds=time.perf_counter() - started)
    except (Exception, KeyboardInterrupt) as error:
        report = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                  "stage": "load_or_verification_exception", "exception_type": type(error).__name__,
                  "scope": "No strict loading or successful API verification is claimed after an exception",
                  "total_seconds": time.perf_counter() - started, "resources": resources()}
        save_report(args.output, report)
        raise
    save_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "checkpoint_sha256", "parameter_count", "checks", "errors")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
