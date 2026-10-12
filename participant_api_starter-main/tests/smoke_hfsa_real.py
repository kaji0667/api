"""Opt-in real-weight HTTP smoke using a temporary synthetic image package.

This checks deployment plumbing, not competition accuracy. Never submits to the
evaluation website, modifies official assets, or prints the private test token.
"""

import argparse
import hashlib
import json
import secrets
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_adapter import RealModel
from server import APIServer, AssetStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target", default="windmill")
    parser.add_argument("--refseg-question", default="The gray small windmill")
    args = parser.parse_args()
    payload = args.image.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    asset_id = "a" * 32

    start = time.perf_counter()
    model = RealModel()
    print(f"Startup: {time.perf_counter() - start:.2f}s", flush=True)
    adapter = model.adapter
    assert adapter.detection.encoder is adapter.counting.encoder
    import torch
    from PIL import Image
    with Image.open(args.image) as source:
        width, height = source.size
        mime_type = Image.MIME[source.format]

    with tempfile.TemporaryDirectory(prefix="hfsa-api-smoke-") as directory:
        package = Path(directory)
        (package / "assets").mkdir()
        (package / "assets" / asset_id).write_bytes(payload)
        manifest = {"format": "rsu-sealed-images-v1", "dataset_id": "synthetic-hfsa-smoke",
                    "assets": [{"asset_id": asset_id, "sha256": digest,
                                "size": len(payload), "mime_type": mime_type}]}
        (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        image_reference = {"asset_id": asset_id, "sha256": digest, "mime_type": mime_type}
        token = secrets.token_urlsafe(32)
        server = APIServer(("127.0.0.1", 0), AssetStore(package), model, token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"

        def call(path, body=None, authorized=True):
            headers = {"Content-Type": "application/json"}
            if authorized:
                headers["Authorization"] = "Bearer " + token
            request = urllib.request.Request(origin + path, headers=headers,
                        data=None if body is None else json.dumps(body).encode("utf-8"))
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                return exc.code, json.load(exc)

        def predict(name, question, constraint, **extra):
            kind = constraint["type"]
            request = {"protocol_version": "2.0", "request_id": f"synthetic:{name}:1", "item_id": name,
                       "question": question, "images": [image_reference],
                       "answer_type": "bbox" if kind == "bbox" else "short_text",
                       "response_constraint": constraint, **extra}
            before = time.perf_counter()
            status, response = call("/v1/predict", request)
            print(f"{name}: HTTP {status}, {time.perf_counter() - before:.2f}s, {response}", flush=True)
            assert status == 200, name
            assert response["request_id"] == request["request_id"] and response["item_id"] == name
            return response["answer"]

        try:
            assert call("/healthz", authorized=False)[0] == 401
            assert call("/healthz") == (200, {"status": "ready", "protocol_version": "2.0",
                                            "dataset_id": "synthetic-hfsa-smoke"})
            classes = adapter.classification.class_names
            choices = {chr(65 + i): name for i, name in enumerate(classes)}
            scene = predict("scene", "What scene category is shown in the image?",
                            {"type": "single_choice", "values": list(choices)}, choices=choices)
            assert scene in choices
            count = predict("count", f"How many {args.target} are there in the image?", {"type": "integer", "minimum": 0})
            assert isinstance(count, str) and count.isdigit()
            presence = predict("presence", f"Is there a {args.target} in the image?",
                               {"type": "enum", "values": ["Yes", "No"]})
            assert presence in {"Yes", "No"}
            bbox_constraint = {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4}
            question = "Return a bounding box. Description: " + args.refseg_question
            box = predict(adapter.grounding_backend, question, bbox_constraint, image_width=width, image_height=height)
            assert 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
            bad = {"protocol_version": "2.0", "request_id": "synthetic:bad:1", "item_id": "bad",
                   "question": "What changed between these images?", "images": [image_reference] * 2,
                   "answer_type": "short_text", "response_constraint": {"type": "enum", "values": ["Yes", "No"]}}
            assert call("/v1/predict", bad) == (500, {"error": "inference_failed"})
            if torch.cuda.is_available():
                print(f"GPU allocated: {torch.cuda.memory_allocated() / 2**30:.2f} GiB; "
                      f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            print("Configured native-task real HTTP smoke passed; shared detection/counting text encoder verified.", flush=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    main()
