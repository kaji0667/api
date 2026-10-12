"""Hash-only delivery audit; never import models or deserialize checkpoints."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


AUTHORIZED_PREVIOUS_SOURCE_CHANGES = frozenset({
    "participant_api_starter-main/hfsa_adapter.py",
    "participant_api_starter-main/ollama_router.py",
    "participant_api_starter-main/tests/test_hfsa_adapter.py",
    "scripts/serve_hfsa_api.sh",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    return {"command": arguments, "exit_code": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def compare(path: Path, expected: str) -> dict:
    current = sha256(path) if path.is_file() else None
    return {"path": str(path), "expected_sha256": expected,
            "current_sha256": current, "matches": current == expected}


def audit(root: Path, *, finalized: bool, previous_inputs: dict | None = None) -> dict:
    receipt_paths = {
        "shared_baseline": root / "runs/vqa/stage1/source_baseline_sha256.json",
        "qformer_delivery": root / "runs/vqa/qformer/reports/delivery_verification.json",
        "qformer_preservation": root / "runs/vqa/qformer/reports/source_integrity.json",
    }
    inputs = {name: {"path": str(path), "sha256": sha256(path)}
              for name, path in receipt_paths.items()}
    for name, item in inputs.items():
        previous = (previous_inputs or {}).get(name, {})
        initial = previous.get("receipt_sha256_at_first_audit", previous.get("sha256", item["sha256"]))
        item["receipt_sha256_at_first_audit"] = initial
        item["matches_initial_audit"] = item["sha256"] == initial
    baseline = read_json(receipt_paths["shared_baseline"])
    delivery = read_json(receipt_paths["qformer_delivery"])
    previous_integrity = read_json(receipt_paths["qformer_preservation"])
    shared = {relative: compare(root / relative, expected)
              for relative, expected in baseline["files"].items()}
    preserved_source, authorized_source = {}, {}
    for relative, expected in delivery["active_source_sha256"].items():
        result = compare(root / relative, expected)
        if relative in AUTHORIZED_PREVIOUS_SOURCE_CHANGES:
            result["change_authorized"] = True
            result["snapshot_finalized"] = finalized
            authorized_source[relative] = result
        else:
            preserved_source[relative] = result

    frozen_assets = {}
    for name, item in previous_integrity["frozen_assets"].items():
        result = compare(Path(item["path"]), item["current_sha256"])
        result["previous_manifest_sha256"] = item["manifest_sha256"]
        result["previous_cache_sha256"] = item["cache_sha256"]
        result["all_previous_hashes_agree"] = (
            result["matches"] and result["current_sha256"] == item["manifest_sha256"] == item["cache_sha256"])
        frozen_assets[name] = result
    historical = {}
    for name, item in previous_integrity["historical_checkpoints"].items():
        checkpoint = Path(item["checkpoint_path"])
        result = compare(checkpoint, item["current_sha256"])
        manifest_path = checkpoint.parent.parent / "manifest.json"
        manifest = read_json(manifest_path)
        result.update({"historical_manifest_path": str(manifest_path),
                       "historical_manifest_sha256": sha256(manifest_path),
                       "current_manifest_checkpoint_sha256": manifest["checkpoint"]["sha256"],
                       "manifest_matches_previous": manifest["checkpoint"]["sha256"] == item["manifest_sha256"],
                       "previous_architecture_id": item["architecture_id"],
                       "deserialized": False})
        historical[name] = result
    qformer_checkpoint = compare(root / "runs/vqa/qformer/weights/vqa_head.pt", delivery["checkpoint_sha256"])
    qformer_manifest = read_json(root / "runs/vqa/qformer/manifest.json")
    qformer_checkpoint["manifest_sha256_matches"] = (
        qformer_manifest["checkpoint"]["sha256"] == delivery["checkpoint_sha256"])
    qformer_checkpoint["deserialized"] = False

    snapshot_files = set((root / "tasks/change").rglob("*.py"))
    snapshot_files.update((root / "third_party/change_agent").glob("*"))
    adjacent = ["hfsa_adapter.py", "ollama_router.py", "model_adapter.py", "server.py",
                "tests/test_hfsa_adapter.py", "tests/test_ollama_router.py", "tests/test_change_adapter.py"]
    snapshot_files.update(root / "participant_api_starter-main" / relative for relative in adjacent)
    snapshot_files.update(root / "scripts" / name for name in (
        "fetch_change_mci.py", "verify_change_api.py", "audit_change_delivery.py", "serve_hfsa_api.sh"))
    snapshot_files.add(root / "tasks/change/README.md")
    snapshot_files = sorted(path for path in snapshot_files if path.is_file())
    source_snapshot = {path.relative_to(root).as_posix(): sha256(path) for path in snapshot_files}
    python_ast = {}
    for path in snapshot_files:
        if path.suffix == ".py":
            relative = path.relative_to(root).as_posix()
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                python_ast[relative] = {"valid": True}
            except (SyntaxError, UnicodeError) as error:
                python_ast[relative] = {"valid": False, "error_type": type(error).__name__}
    shell_syntax = {}
    for relative in ("scripts/prepare_vqa.sh", "scripts/train_vqa.sh", "scripts/eval_vqa.sh",
                     "scripts/serve_hfsa_api.sh"):
        result = command(["bash", "-n", str(root / relative)])
        result["valid"] = result["exit_code"] == 0
        shell_syntax[relative] = result
    git_roots = {}
    for directory in (root.parents[1], root):
        git_roots[str(directory)] = {
            "rev_parse": command(["git", "-C", str(directory), "rev-parse", "--show-toplevel"]),
            "status": command(["git", "-C", str(directory), "status", "--short"]),
        }
    for name, path in receipt_paths.items():
        inputs[name]["sha256_after_audit"] = sha256(path)
        inputs[name]["unchanged_during_audit"] = inputs[name]["sha256"] == inputs[name]["sha256_after_audit"]
    checks = {
        "all_16_shared_source_hashes_preserved": len(shared) == 16 and all(item["matches"] for item in shared.values()),
        "all_unauthorized_delivery_source_hashes_preserved": all(item["matches"] for item in preserved_source.values()),
        "qformer_checkpoint_preserved": qformer_checkpoint["matches"] and qformer_checkpoint["manifest_sha256_matches"],
        "frozen_assets_preserved": all(item["all_previous_hashes_agree"] for item in frozen_assets.values()),
        "historical_checkpoints_preserved": all(item["matches"] and item["manifest_matches_previous"]
                                                  for item in historical.values()),
        "historical_reports_untouched": all(item["unchanged_during_audit"] and item["matches_initial_audit"]
                                              for item in inputs.values()),
        "python_ast_passed": all(item["valid"] for item in python_ast.values()),
        "bash_syntax_passed": all(item["valid"] for item in shell_syntax.values()),
    }
    return {
        "schema_version": 1, "captured_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "finalized" if finalized else "pending_final_authorized_source_snapshot",
        "overall_integrity_and_syntax_passed": all(checks.values()), "checks": checks,
        "scope": "File hashing/JSON metadata/Python AST/bash -n/read-only Git checks only; no model import, checkpoint deserialization, private official requests, credentials, process environments, service operations, old-report mutations or Git initialization/push",
        "input_receipts": inputs,
        "shared_source_baseline": {"count": len(shared), "description": baseline["description"], "results": shared},
        "qformer_delivery_preserved_source": {"count": len(preserved_source), "results": preserved_source},
        "authorized_previous_source_changes": {"count": len(authorized_source), "results": authorized_source},
        "qformer_checkpoint": qformer_checkpoint, "frozen_assets": frozen_assets,
        "historical_checkpoints": historical,
        "python_ast": {"count": len(python_ast), "results": python_ast},
        "shell_syntax": {"count": len(shell_syntax), "results": shell_syntax},
        "current_source_sha256": source_snapshot,
        "source_snapshot_finalized": finalized,
        "pending_updated_fields": [] if finalized else ["authorized_previous_source_changes", "current_source_sha256"],
        "git_roots": git_roots,
        "git_action": "Read-only checks only; neither repository initialized, repaired, committed or pushed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--finalize", action="store_true", help="Use only after all authorized source edits finish")
    args = parser.parse_args()
    root = args.code_root.expanduser().resolve()
    report_path = args.report or root / "runs/change/mci/reports/source_integrity.json"
    previous = read_json(report_path) if report_path.is_file() else {}
    result = audit(root, finalized=args.finalize, previous_inputs=previous.get("input_receipts"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "status": result["audit_status"],
                      "passed": result["overall_integrity_and_syntax_passed"], "checks": result["checks"]}))
    if not result["overall_integrity_and_syntax_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
