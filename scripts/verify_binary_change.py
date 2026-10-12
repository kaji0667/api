"""Independently verify the binary model against fixed public author demos.

The original author's preprocessing and full V6 model compute the reference.
Public labels enter independent scoring only, after each image prediction.
No production service, official image or evaluation website is accessed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import types

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from fetch_binary_change import (BASE_NAMES, NETWORK_NAMES, PUBLIC_HASHES,
                                 REVISION, SAMPLES, SOURCE_HASHES, sha256)


def source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def author_model(source_root: Path):
    """Execute the original complete model with only unused import adaptation."""
    namespace = "author_changeformer_models"
    package = types.ModuleType(namespace)
    package.__path__ = [str(source_root / "models")]
    sys.modules[namespace] = package
    source = (source_root / "models/ChangeFormer.py").read_text(encoding="utf-8")
    replacements = {
        "from models.ChangeFormerBaseNetworks import *":
            f"from {namespace}.ChangeFormerBaseNetworks import *",
        "from models.pixel_shuffel_up import PS_UP":
            f"from {namespace}.pixel_shuffel_up import PS_UP",
        "from models.help_funcs import TwoLayerConv2d, save_to_mat":
            "# V6 does not reference these helpers; their unused einops import is omitted.",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("Original author import layout changed")
        source = source.replace(before, after)
    module = types.ModuleType(namespace + ".ChangeFormer")
    module.__file__ = str(source_root / "models/ChangeFormer.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.ChangeFormerV6, replacements


def definition_asts(path: Path) -> dict:
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))}


def independent_regions(mask):
    """Independent integral rectangle counting; no deployment helper used."""
    import numpy as np

    height, width = mask.shape
    prefix = np.pad(np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1), ((1, 0), (1, 0)))
    def pixels(top, left, bottom, right):
        return int(prefix[bottom, right] - prefix[top, right] - prefix[bottom, left] + prefix[top, left])
    row_edges = [0, height // 3, 2 * height // 3, height]
    col_edges = [0, width // 3, 2 * width // 3, width]
    names = ("top_left", "top_center", "top_right", "middle_left", "center", "middle_right",
             "bottom_left", "bottom_center", "bottom_right")
    result = {}
    for index, name in enumerate(names):
        row, col = divmod(index, 3)
        result[name] = pixels(row_edges[row], col_edges[col], row_edges[row + 1], col_edges[col + 1])
    result.update({"upper_left": pixels(0, 0, height // 2, width // 2),
                   "upper_right": pixels(0, width // 2, height // 2, width),
                   "lower_left": pixels(height // 2, 0, height, width // 2),
                   "lower_right": pixels(height // 2, width // 2, height, width)})
    return result


def winner(pixels: dict, names: tuple[str, ...]) -> str | None:
    maximum = max(pixels[name] for name in names)
    winners = [name for name in names if pixels[name] == maximum]
    return winners[0] if maximum > 0 and len(winners) == 1 else None


def validate_public_manifest(public: dict) -> None:
    expected_rows = [{"id": name.removesuffix(".png"), "image_a": "public/A/" + name,
                      "image_b": "public/B/" + name, "label": "public/label/" + name}
                     for name in SAMPLES]
    if (not isinstance(public, dict) or public.get("public_only") is not True
            or public.get("source_revision") != REVISION
            or public.get("samples") != expected_rows):
        raise ValueError("Only exact fixed author A/B/GT demo rows are allowed")


def verify_preflight(bundle: Path) -> dict:
    """Repeat only manifest/source checks, including negative GT-path fixtures."""
    public = json.loads((bundle / "public_manifest.json").read_text(encoding="utf-8"))
    validate_public_manifest(public)
    checks = {"exact_fixed_public_ab_gt_rows": True}
    for name, key, value in (("gt_as_image_rejected", "image_a", "public/label/0_2.png"),
                             ("image_order_swap_rejected", "image_a", "public/B/0_2.png"),
                             ("path_traversal_rejected", "image_b", "../private.png"),
                             ("foreign_gt_rejected", "label", "public/label/3_4.png")):
        changed = copy.deepcopy(public)
        changed["samples"][0][key] = value
        try:
            validate_public_manifest(changed)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    for name, mutate in (("row_order_change_rejected", lambda rows: rows.reverse()),
                          ("extra_row_rejected", lambda rows: rows.append(dict(rows[0]))),
                          ("extra_row_field_rejected", lambda rows: rows[0].update(extra="GT must not be an input"))):
        changed = copy.deepcopy(public)
        mutate(changed["samples"])
        try:
            validate_public_manifest(changed)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    checks["fixed_public_file_hashes_unchanged"] = all(
        sha256(bundle / "public" / relative) == expected for relative, expected in PUBLIC_HASHES.items())
    checks["fixed_author_source_hashes_unchanged"] = all(
        sha256(bundle / "source" / relative) == expected for relative, expected in SOURCE_HASHES.items())
    non256 = verify_non256_preprocessing(bundle)
    checks["non256_author_preprocessing_exact"] = non256["exact_tensor_parity"]
    return {"report_type": "binary_public_preflight_recheck", "checks": checks,
            "status": "passed" if all(checks.values()) else "failed",
            "no_model_prediction_in_this_recheck": True,
            "verification_source_sha256": sha256(Path(__file__)),
            "binary_predictor_source_sha256": sha256(CODE_DIR / "tasks/change/binary_change.py"),
            "public_manifest_sha256": sha256(bundle / "public_manifest.json"),
            "non256_preprocessing": non256}


def verify_non256_preprocessing(bundle: Path) -> dict:
    """Check resized rectangular public images against real author eval tensors."""
    import numpy as np
    from PIL import Image
    from tasks.change.binary_change import read_binary_pair

    author_aug = source_module("author_binary_resize_data_utils", bundle / "source/datasets/data_utils.py")
    with tempfile.TemporaryDirectory(prefix="hfsa-public-binary-preprocessing-") as directory:
        paths, rgb_arrays = [], []
        for part in ("A", "B"):
            source_path = bundle / "public" / part / SAMPLES[0]
            with Image.open(source_path) as image:
                rectangular = image.convert("RGB").resize((319, 287), Image.Resampling.BICUBIC)
            path = Path(directory) / (part + ".png")
            rectangular.save(path)
            paths.append(path)
            rgb_arrays.append(np.asarray(rectangular))
        first, second, original_hw = read_binary_pair(*paths)
        expected, _ = author_aug.CDDataAugmentation(img_size=256).transform(rgb_arrays, [], to_tensor=True)
        errors = [float((actual - reference.unsqueeze(0)).abs().max())
                  for actual, reference in zip((first, second), expected)]
        return {"public_source_sample": SAMPLES[0], "generated_width_height": [319, 287],
                "generation": "Public A/B each converted RGB and resized PIL bicubic, fixed before this check",
                "original_hw": list(original_hw), "tensor_shapes": [list(first.shape), list(second.shape)],
                "input_max_errors": errors,
                "exact_tensor_parity": errors == [0., 0.] and original_hw == (287, 319),
                "author_eval_called_with_labels": False, "no_model_forward": True}


def verify(bundle: Path) -> dict:
    import numpy as np
    from PIL import Image
    import torch
    from tasks.change.binary_change import BinaryChangePredictor, read_binary_pair

    started = time.monotonic()
    public = json.loads((bundle / "public_manifest.json").read_text(encoding="utf-8"))
    validate_public_manifest(public)
    source_root = bundle / "source"
    for relative, expected in SOURCE_HASHES.items():
        if sha256(source_root / relative) != expected:
            raise ValueError("Original author source integrity mismatch")
    for relative, expected in PUBLIC_HASHES.items():
        if sha256(bundle / "public" / relative) != expected:
            raise ValueError("Fixed author public sample integrity mismatch")
    audit = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if not audit.get("serialization_audit", {}).get("pure_state_reloaded_without_safe_globals"):
        raise ValueError("Missing safe checkpoint conversion evidence")
    predictor = BinaryChangePredictor(bundle, device="cpu")
    constructed = time.monotonic() - started
    author_class, import_adaptations = author_model(source_root)
    with torch.random.fork_rng(devices=[]):
        reference = author_class(input_nc=3, output_nc=2, embed_dim=256, decoder_softmax=False)
    state = torch.load(bundle / audit["checkpoint"]["path"], map_location="cpu", weights_only=True)
    reference.load_state_dict(state, strict=True)
    reference.requires_grad_(False).eval()
    author_aug = source_module("author_changeformer_data_utils", source_root / "datasets/data_utils.py")
    original_defs = {**definition_asts(source_root / "models/ChangeFormerBaseNetworks.py"),
                     **definition_asts(source_root / "models/ChangeFormer.py")}
    deployed_defs = definition_asts(CODE_DIR / "tasks/change/_binary/network.py")
    unchanged = {name: deployed_defs[name] == original_defs[name] for name in (*BASE_NAMES, *NETWORK_NAMES)}
    counts_equal = set(reference.state_dict()) == set(predictor.model.state_dict()) == set(state)
    report = {"report_type": "public_dsifn_binary_change_verification", "scope": "Four fixed author demos, not a complete independent test set or official accuracy",
              "architecture_id": audit["architecture_id"], "source_revision": REVISION,
              "checkpoint": audit["checkpoint"], "loading": predictor.loading,
              "author_import_adaptations_only": import_adaptations,
              "serialization_audit": audit["serialization_audit"],
              "public_manifest_sha256": sha256(bundle / "public_manifest.json"),
              "fixed_sample_selection": public["selection_rule"],
              "ground_truth_role": "Scoring only, never a model input or preprocessing argument",
              "model_construct_seconds": constructed,
              "checks": {"all_15_author_definitions_ast_unchanged": all(unchanged.values()),
                         "exact_fixed_public_ab_gt_rows": True,
                         "strict_full_state_key_coverage": counts_equal,
                         "pure_state_loaded_with_weights_only": True},
              "definition_ast_parity": unchanged, "samples": []}
    four = ("upper_left", "upper_right", "lower_left", "lower_right")
    nine = ("top_left", "top_center", "top_right", "middle_left", "center", "middle_right",
            "bottom_left", "bottom_center", "bottom_right")
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    with torch.inference_mode():
        for row in public["samples"]:
            first_path, second_path = bundle / row["image_a"], bundle / row["image_b"]
            first, second, original_hw = read_binary_pair(first_path, second_path)
            with Image.open(first_path) as image:
                rgb_a = np.asarray(image.convert("RGB"))
            with Image.open(second_path) as image:
                rgb_b = np.asarray(image.convert("RGB"))
            # Invoke the author's real evaluation augmentation, without labels.
            reference_inputs, _ = author_aug.CDDataAugmentation(img_size=256).transform([rgb_a, rgb_b], [], to_tensor=True)
            input_errors = [float((actual - expected.unsqueeze(0)).abs().max())
                            for actual, expected in zip((first, second), reference_inputs)]
            original_outputs = reference(*(value.unsqueeze(0) for value in reference_inputs))
            deployed_outputs = predictor.model(first, second)
            logit_errors = [float((actual - expected).abs().max())
                            for actual, expected in zip(deployed_outputs, original_outputs)]
            native_mask = original_outputs[-1].argmax(dim=1)[0].numpy().astype(np.uint8)
            actual = predictor.evidence(first_path, second_path)
            summary = actual.summary()
            actual_regions = summary["mask_grid"]["binary"]["region_pixels"]
            independent_actual = independent_regions(actual.mask)
            # Only now read GT for independent binary and region scoring.
            with Image.open(bundle / row["label"]) as label_image:
                label = np.asarray(label_image, dtype=np.uint8)
            if label.shape != (256, 256) or not set(np.unique(label)).issubset({0, 255}):
                raise ValueError("Unexpected author's demo label shape/classes")
            truth = label // 255
            tp = int(np.logical_and(actual.mask == 1, truth == 1).sum())
            fp = int(np.logical_and(actual.mask == 1, truth == 0).sum())
            fn = int(np.logical_and(actual.mask == 0, truth == 1).sum())
            tn = int(np.logical_and(actual.mask == 0, truth == 0).sum())
            for key, value in {"tp": tp, "fp": fp, "fn": fn, "tn": tn}.items():
                totals[key] += value
            gt_regions = independent_regions(truth)
            scored = {"id": row["id"], "original_hw": list(original_hw),
                      "input_max_errors": input_errors, "all_scale_logit_max_errors": logit_errors,
                      "mask_equals_author": bool(np.array_equal(actual.mask, native_mask)),
                      "region_counts_equal_independent_integral": actual_regions == independent_actual,
                      "elapsed_seconds": actual.elapsed_seconds, "binary_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                      "predicted_changed_pixels": int(actual.mask.sum()), "gt_changed_pixels": int(truth.sum()),
                      "foreground_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                      "predicted_regions": actual_regions, "gt_regions": gt_regions}
            for partition, names in (("four", four), ("nine", nine)):
                predicted, gold = winner(actual_regions, names), winner(gt_regions, names)
                scored[partition + "_region"] = {"predicted_unique_positive_max": predicted,
                                                "gt_unique_positive_max": gold,
                                                "correct_if_gt_unique": predicted == gold if gold is not None else None}
            report["samples"].append(scored)
            print(json.dumps({"stage": "public_binary_sample", "id": row["id"],
                              "mask_equals_author": scored["mask_equals_author"],
                              "foreground_iou": scored["foreground_iou"]}), flush=True)
        first_row = public["samples"][0]
        same = predictor.evidence(bundle / first_row["image_a"], bundle / first_row["image_a"])
        reverse = predictor.evidence(bundle / first_row["image_b"], bundle / first_row["image_a"])
        report["same_image_control"] = {"predicted_changed_pixels": int(same.mask.sum()),
                                        "expected_changed_pixels_by_construction": 0,
                                        "correct_no_change": not bool(same.mask.any())}
        report["reverse_order_diagnostic"] = {"predicted_changed_pixels": int(reverse.mask.sum()),
            "direction_labels_available": False, "scored_against_forward_gt": False}
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    report["binary_diagnostic"] = {"n": len(report["samples"]), "counts": totals,
        "foreground_iou": tp / (tp + fp + fn), "foreground_f1": 2 * tp / (2 * tp + fp + fn),
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn)}
    report["region_diagnostic"] = {}
    for partition in ("four", "nine"):
        rows = [row[partition + "_region"] for row in report["samples"]]
        eligible = [row for row in rows if row["gt_unique_positive_max"] is not None]
        report["region_diagnostic"][partition] = {"n": len(rows), "gt_unique_positive": len(eligible),
            "correct": sum(row["correct_if_gt_unique"] for row in eligible),
            "refused_no_unique_prediction": sum(row["predicted_unique_positive_max"] is None for row in eligible)}
    report["checks"].update({"all_author_input_tensors_exact": all(max(row["input_max_errors"]) == 0 for row in report["samples"]),
        "all_five_author_logits_exact": all(max(row["all_scale_logit_max_errors"]) == 0 for row in report["samples"]),
        "all_masks_equal_author": all(row["mask_equals_author"] for row in report["samples"]),
        "all_region_counts_independently_equal": all(row["region_counts_equal_independent_integral"] for row in report["samples"])})
    report["elapsed_seconds"] = time.monotonic() - started
    report["verification_source_sha256"] = sha256(Path(__file__))
    report["binary_predictor_source_sha256"] = sha256(CODE_DIR / "tasks/change/binary_change.py")
    report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["rss_scope"] = "Verification process with two model instances, not production incremental RSS"
    report["status"] = "passed" if all(report["checks"].values()) else "failed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=CODE_DIR / "runs/change/changeformer_dsifn")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="No model load; verify fixed paths and negative GT input fixtures")
    args = parser.parse_args()
    bundle = args.bundle_dir.resolve()
    filename = "preflight_verification.json" if args.preflight_only else "public_verification.json"
    output = args.output or bundle / "reports" / filename
    report = verify_preflight(bundle) if args.preflight_only else verify(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    keys = ("status", "checks") if args.preflight_only else ("status", "checks", "binary_diagnostic", "region_diagnostic",
                                                             "same_image_control", "elapsed_seconds", "peak_process_rss_bytes")
    print(json.dumps({key: report[key] for key in keys}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
