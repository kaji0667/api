"""Check a running participant API with one synthetic (unscored) request."""

import argparse
import json
import os
import urllib.request
from pathlib import Path


def call(url: str, token: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=130) as response:
        if response.headers.get_content_type() != "application/json":
            raise ValueError(f"expected JSON, got {response.headers.get_content_type()}")
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, help="http://127.0.0.1:9001 or public HTTPS origin")
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get("MODEL_API_KEY", "")
    if not token:
        parser.error("MODEL_API_KEY is required")
    origin = args.origin.rstrip("/")
    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    health = call(origin + "/healthz", token)
    if health.get("status") != "ready" or health.get("protocol_version") != "2.0":
        raise ValueError(f"health check failed: {health}")
    if health.get("dataset_id") != manifest["dataset_id"]:
        raise ValueError("API dataset_id does not match local manifest")
    first = manifest["assets"][0]
    payload = {
        "protocol_version": "2.0",
        "request_id": "synthetic-check:1",
        "item_id": "SYNTHETIC_CHECK",
        "question": "Is there anything in the image? Answer Yes or No.",
        "images": [{key: first[key] for key in ("asset_id", "sha256", "mime_type")}],
        "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
        "answer_type": "short_text",
    }
    result = call(origin + "/v1/predict", token, payload)
    if any(result.get(key) != payload[key] for key in ("protocol_version", "request_id", "item_id")):
        raise ValueError("prediction response did not echo protocol/request/item identifiers")
    if not isinstance(result.get("answer"), str):
        raise ValueError("prediction answer must be a string for this check")
    print(f"OK: HTTPS/API handshake, dataset {health['dataset_id']}, one unscored prediction")


if __name__ == "__main__":
    main()
