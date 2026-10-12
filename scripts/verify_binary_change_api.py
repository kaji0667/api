"""Exercise real binary inference and local routing through temporary HTTP.

Uses fixed public author demos and synthetic questions. No production process,
credential or official evaluation is accessed. Every HTTP request recomputes
visual evidence; expected answers are independently counted after prediction.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "participant_api_starter-main"))

from fetch_binary_change import PUBLIC_HASHES, REVISION, SAMPLES, sha256
from verify_binary_change import independent_regions, winner


def verify(bundle: Path) -> dict:
    from hfsa_adapter import HFSAAdapter
    from server import APIServer, AssetStore
    from tasks.change.answering import EvidenceAnswerer
    from tasks.change.binary_change import BinaryChangePredictor
    from tasks.change.inference import ChangePredictor

    public = json.loads((bundle / "public_manifest.json").read_text(encoding="utf-8"))
    expected_rows = [{"id": name.removesuffix(".png"), "image_a": f"public/A/{name}",
                      "image_b": f"public/B/{name}", "label": f"public/label/{name}"} for name in SAMPLES]
    if (public.get("public_only") is not True or public.get("source_revision") != REVISION
            or public.get("samples") != expected_rows):
        raise ValueError("Expected only fixed public author demo inputs")
    for relative, digest in PUBLIC_HASHES.items():
        if sha256(bundle / "public" / relative) != digest:
            raise ValueError("Public input integrity mismatch")
    binary = BinaryChangePredictor(bundle, device="cpu")
    events = []
    original_evidence = binary.evidence

    def recorded_evidence(before, after):
        evidence = original_evidence(before, after)
        events.append(evidence)
        return evidence

    binary.evidence = recorded_evidence
    # Test the production binary dispatch without duplicating the unrelated MCI
    # model in memory. The factory's eager loading is tested separately.
    change = ChangePredictor.__new__(ChangePredictor)
    change.binary_predictor = binary
    change.answerer = EvidenceAnswerer()

    def forbidden_mci(*args, **kwargs):
        raise AssertionError("Generic binary query reached MCI")

    change.evidence = forbidden_mci
    adapter = HFSAAdapter(None, None, None, None, change=change)
    report = {"status": "running", "public_only": True, "official_material_used": False,
              "real_local_task_router": True, "visual_cache_used": False,
              "model_loading": binary.loading, "requests": []}
    four = ("upper_left", "upper_right", "lower_left", "lower_right")
    nine = ("top_left", "top_center", "top_right", "middle_left", "center", "middle_right",
            "bottom_left", "bottom_center", "bottom_right")
    nine_values = ("top_left", "top", "top_right", "left", "center", "right",
                   "bottom_left", "bottom", "bottom_right")
    with tempfile.TemporaryDirectory(prefix="hfsa-public-binary-") as temporary:
        root = Path(temporary)
        (root / "assets").mkdir()
        entries, references = {}, {}
        for row in public["samples"]:
            for field in ("image_a", "image_b"):
                relative = row[field]
                source = (bundle / relative).resolve()
                if not source.is_relative_to(bundle.resolve()):
                    raise ValueError("Public asset outside bundle")
                payload = source.read_bytes()
                digest = sha256(source)
                asset_id = digest[:32]
                (root / "assets" / asset_id).write_bytes(payload)
                references[relative] = {"asset_id": asset_id, "sha256": digest, "mime_type": "image/png"}
                entries[asset_id] = {**references[relative], "size": len(payload)}
        (root / "manifest.json").write_text(json.dumps({"format": "rsu-sealed-images-v1",
            "dataset_id": "public-changeformer-demo", "assets": list(entries.values())}), encoding="utf-8")
        key = secrets.token_urlsafe(32)
        server = APIServer(("127.0.0.1", 0), AssetStore(root), adapter, key)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with patch.dict("os.environ", {"HFSA_API_REPLAY_DIR": "", "HFSA_API_DIAGNOSTIC_FILE": ""}), \
                    redirect_stderr(io.StringIO()):
                for row in public["samples"]:
                    for partition, names in (("nine", nine), ("four", four)):
                        index = len(report["requests"])
                        values = list(nine_values) if partition == "nine" else ["q3", "q1", "q4", "q2"]
                        labels = dict(zip(values, names)) if partition == "nine" else {
                            "q3": "lower_left", "q1": "upper_left", "q4": "lower_right", "q2": "upper_right"}
                        request = {"protocol_version": "2.0", "request_id": f"public-binary:{index}:1",
                            "item_id": f"PUBLIC_BINARY_{index}",
                            "question": "Which region has the largest change area?" if partition == "nine"
                                        else "Which quadrant has changed the most?",
                            "answer_type": "region_label" if partition == "nine" else "single_choice",
                            "images": [references[row["image_a"]], references[row["image_b"]]],
                            "response_constraint": {"type": "enum" if partition == "nine" else "single_choice",
                                                    "values": values, "question_form": "change_region"}}
                        if partition == "four":
                            request["choices"] = labels
                        call = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/predict",
                            data=json.dumps(request).encode(), headers={"Content-Type": "application/json",
                                                                       "Authorization": "Bearer " + key})
                        before_calls, started = len(events), time.monotonic()
                        try:
                            with opener.open(call, timeout=60) as response:
                                status, result = response.status, json.load(response)
                        except urllib.error.HTTPError as error:
                            status, result = error.code, json.load(error)
                        visual_calls = len(events) - before_calls
                        expected = None
                        if visual_calls == 1:
                            chosen = winner(independent_regions(events[-1].mask), names)
                            expected = next((value for value, name in labels.items() if name == chosen), None)
                        expected_response = ({"protocol_version": "2.0", "request_id": request["request_id"],
                            "item_id": request["item_id"], "answer": expected} if expected is not None
                            else {"error": "inference_failed"})
                        passed = visual_calls == 1 and status == (200 if expected is not None else 500) and result == expected_response
                        report["requests"].append({"sample_id": row["id"], "partition": partition,
                            "http_status": status, "visual_calls": visual_calls, "passed": passed,
                            "elapsed_seconds": time.monotonic() - started,
                            "visual_seconds": events[-1].elapsed_seconds if visual_calls == 1 else None})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    report["status"] = "passed" if len(report["requests"]) == 8 and all(row["passed"] for row in report["requests"]) else "failed"
    report["scope"] = "Answer-to-predicted-mask consistency through real HTTP/router; GT accuracy is reported by verify_binary_change.py"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=CODE_DIR / "runs/change/changeformer_dsifn")
    args = parser.parse_args()
    report = verify(args.bundle.resolve())
    output = args.bundle / "reports/public_api_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        report["previous_attempts"] = prior.get("previous_attempts", [])
        if prior.get("status") != "passed":
            report["previous_attempts"].append({key: value for key, value in prior.items() if key != "previous_attempts"})
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "requests": len(report["requests"]),
                      "passed": sum(row["passed"] for row in report["requests"])}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
