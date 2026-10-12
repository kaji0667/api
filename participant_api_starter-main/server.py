"""Small, dependency-free participant API for the local-image protocol 2.0."""

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from model_adapter import build_model


VERSION = "2.0"
MAX_BODY = 128 * 1024
MAX_ASSET_SIZE = 256 * 1024 * 1024
ASSET_ID = re.compile(r"[0-9a-f]{32}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


class APIError(Exception):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code


class AssetStore:
    def __init__(self, package: Path):
        manifest_path = package / "manifest.json"
        asset_dir = package / "assets"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("manifest.json must be a regular file")
        if manifest_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("manifest.json is too large")
        if asset_dir.is_symlink() or not asset_dir.is_dir():
            raise ValueError("assets must be a real directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != {"format", "dataset_id", "assets"}:
            raise ValueError("invalid manifest schema")
        if manifest["format"] != "rsu-sealed-images-v1":
            raise ValueError("unsupported manifest format")
        dataset_id = manifest["dataset_id"]
        if not isinstance(dataset_id, str) or not dataset_id or len(dataset_id) > 128:
            raise ValueError("invalid dataset_id")
        if not isinstance(manifest["assets"], list) or not manifest["assets"]:
            raise ValueError("manifest has no assets")
        entries: dict[str, dict[str, Any]] = {}
        for entry in manifest["assets"]:
            if not isinstance(entry, dict) or set(entry) != {"asset_id", "sha256", "size", "mime_type"}:
                raise ValueError("invalid manifest asset entry")
            asset_id, digest, size, mime_type = (
                entry["asset_id"], entry["sha256"], entry["size"], entry["mime_type"]
            )
            if not isinstance(asset_id, str) or not ASSET_ID.fullmatch(asset_id) or asset_id in entries:
                raise ValueError("invalid or duplicate asset_id")
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise ValueError("invalid asset sha256")
            if type(size) is not int or not 0 < size <= MAX_ASSET_SIZE or mime_type not in MIME_TYPES:
                raise ValueError("invalid asset size or mime_type")
            info = (asset_dir / asset_id).lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise ValueError(f"asset missing or wrong size: {asset_id}")
            entries[asset_id] = entry
        self.asset_dir = asset_dir
        self.dataset_id = dataset_id
        self.entries = entries

    def verified_path(self, spec: Any) -> Path:
        if not isinstance(spec, dict) or set(spec) != {"asset_id", "sha256", "mime_type"}:
            raise APIError(422, "invalid_image_reference")
        asset_id = spec["asset_id"]
        if not isinstance(asset_id, str) or not ASSET_ID.fullmatch(asset_id):
            raise APIError(422, "invalid_image_reference")
        entry = self.entries.get(asset_id)
        if entry is None or any(entry[key] != spec[key] for key in ("sha256", "mime_type")):
            raise APIError(422, "asset_metadata_mismatch")
        path = self.asset_dir / asset_id
        digest = hashlib.sha256()
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size != entry["size"]:
                    raise APIError(422, "asset_size_mismatch")
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise APIError(422, "asset_unavailable") from exc
        if digest.hexdigest() != entry["sha256"]:
            raise APIError(422, "asset_sha256_mismatch")
        return path


def validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("protocol_version") != VERSION:
        raise APIError(400, "invalid_protocol")
    for key in ("request_id", "item_id", "question"):
        if not isinstance(value.get(key), str) or not value[key] or len(value[key]) > 16384:
            raise APIError(400, f"invalid_{key}")
    images = value.get("images")
    if not isinstance(images, list) or not 1 <= len(images) <= 2:
        raise APIError(400, "invalid_images")
    constraint = value.get("response_constraint")
    if not isinstance(constraint, dict) or constraint.get("type") not in {
        "enum", "single_choice", "short_text", "integer", "bbox"
    }:
        raise APIError(400, "invalid_response_constraint")
    if constraint["type"] in {"enum", "single_choice"}:
        values = constraint.get("values")
        if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
            raise APIError(400, "invalid_response_constraint")
    if constraint["type"] == "bbox":
        if any(type(value.get(key)) is not int or value[key] <= 0 for key in ("image_width", "image_height")):
            raise APIError(400, "invalid_image_dimensions")
    return value


def validate_answer(request: dict[str, Any], answer: Any) -> None:
    if request.get("answer_type") == "bbox":
        if not (
            isinstance(answer, list) and len(answer) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in answer)
            and answer[0] < answer[2] and answer[1] < answer[3]
        ):
            raise APIError(502, "invalid_model_answer")
    elif not isinstance(answer, str) or len(answer) > 4096:
        raise APIError(502, "invalid_model_answer")


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], assets: AssetStore, model: Any, token: str):
        super().__init__(address, APIHandler)
        self.assets = assets
        self.model = model
        self.token = token
        self.inference_slots = threading.BoundedSemaphore(1)


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Deliberately do not log URL query strings, request bodies, or credentials.
        pass

    def _json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _authorized(self) -> bool:
        expected = "Bearer " + self.server.token
        if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
            self._json(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._json(404, {"error": "not_found"})
        elif self._authorized():
            self._json(200, {
                "status": "ready", "protocol_version": VERSION,
                "dataset_id": self.server.assets.dataset_id,
            })

    def do_POST(self) -> None:
        if self.path != "/v1/predict":
            self._json(404, {"error": "not_found"})
            return
        if not self._authorized():
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "json_required"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            self._json(413, {"error": "invalid_request_size"})
            return
        if not self.server.inference_slots.acquire(blocking=False):
            self._json(503, {"error": "model_busy"})
            return
        try:
            try:
                request = validate_request(json.loads(self.rfile.read(length)))
                image_paths = [self.server.assets.verified_path(image) for image in request["images"]]
                answer = self.server.model.predict(request, image_paths)
                validate_answer(request, answer)
                self._json(200, {
                    "protocol_version": VERSION, "request_id": request["request_id"],
                    "item_id": request["item_id"], "answer": answer,
                })
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(400, {"error": "invalid_json"})
            except APIError as exc:
                self._json(exc.status, {"error": exc.code})
            except Exception:
                # Never echo a traceback, model prompt, local path, or secret to the caller.
                print("model inference failed; inspect local logs", file=sys.stderr)
                self._json(500, {"error": "inference_failed"})
        finally:
            self.server.inference_slots.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="directory containing manifest.json and assets/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()
    token = os.environ.get("MODEL_API_KEY", "")
    if len(token) < 32:
        parser.error("set MODEL_API_KEY to a random secret of at least 32 characters")
    assets = AssetStore(args.package)
    model = build_model()
    if os.environ.get("MODEL_MODE", "demo") == "demo":
        print("WARNING: demo mode only checks the API; answers have no scoring value", file=sys.stderr)
    with APIServer((args.host, args.port), assets, model, token) as server:
        print(f"API listening on {args.host}:{args.port}; dataset_id={assets.dataset_id}")
        server.serve_forever(poll_interval=0.2)


if __name__ == "__main__":
    main()
