"""Dependency-free local web shell for the HFSA task router."""

from __future__ import annotations

import argparse
import json
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tasks.routing import (
    TaskInputError,
    TaskInterfaceUnavailableError,
    TaskRouter,
    UnknownTaskError,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/earth-texture-realistic.png": ("earth-texture-realistic.png", "image/png"),
    "/earth-rotation.mp4": ("earth-rotation.mp4", "video/mp4"),
    "/earth-rotation-poster.jpg": ("earth-rotation-poster.jpg", "image/jpeg"),
    "/satellite-realistic.png": ("satellite-realistic.png", "image/png"),
}
MAX_REQUEST_BYTES = 64 * 1024 * 1024


def create_default_router() -> TaskRouter:
    """Register all task adapters without importing model stacks at startup."""

    router = TaskRouter()

    def classification_factory(config):
        from tasks.routing.adapters.classification import SceneClassificationAdapter

        return SceneClassificationAdapter(config)

    def detection_factory(config):
        from tasks.routing.adapters.detection import DetectionAdapter

        return DetectionAdapter(config)

    def counting_factory(config):
        from tasks.routing.adapters.counting import CountingAdapter

        return CountingAdapter(config)

    def refseg_factory(config):
        from tasks.routing.adapters.refseg import RefSegAdapter

        return RefSegAdapter(config)

    router.register_adapter("classification", classification_factory)
    router.register_adapter("detection", detection_factory)
    router.register_adapter("counting", counting_factory)
    router.register_adapter("refseg", refseg_factory)
    return router


class HFSAWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, router: TaskRouter, web_root: Path) -> None:
        self.router = router
        self.web_root = web_root.resolve()
        super().__init__(server_address, HFSARequestHandler)

    def server_close(self) -> None:
        self.router.close()
        super().server_close()


class HFSARequestHandler(BaseHTTPRequestHandler):
    server: HFSAWebServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError("Request body must be a non-empty JSON document under 64 MiB.")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object.")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/favicon.ico":
            self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return
        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "service": "remote-sensing-multitask-web"},
            )
            return
        if path == "/api/tasks":
            self._send_json(HTTPStatus.OK, {"tasks": self.server.router.list_tasks()})
            return
        asset = ASSETS.get(path)
        if asset is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        filename, content_type = asset
        asset_path = (self.server.web_root / filename).resolve()
        if asset_path.parent != self.server.web_root or not asset_path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Missing web asset: {filename}"})
            return
        self._send_bytes(HTTPStatus.OK, asset_path.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            payload = self._read_json()
            task_id = str(payload.get("task_id", ""))
            inputs = payload.get("inputs", {})
            result = self.server.router.predict(task_id, inputs)
        except (ValueError, TaskInputError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)})
            return
        except UnknownTaskError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"success": False, "error": str(exc)})
            return
        except TaskInterfaceUnavailableError as exc:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"success": False, "status": "interface_pending", "message": str(exc)},
            )
            return
        except Exception as exc:  # noqa: BLE001 - HTTP boundary must return structured failures.
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "success": False,
                    "status": "inference_error",
                    "message": f"模型推理失败：{exc}",
                },
            )
            return
        self._send_json(HTTPStatus.OK, {"success": True, "task_id": task_id, "result": result})


def create_server(host: str = "127.0.0.1", port: int = 7860, router: TaskRouter | None = None) -> HFSAWebServer:
    if not 0 <= int(port) <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if not WEB_ROOT.is_dir():
        raise FileNotFoundError(f"HFSA web asset directory not found: {WEB_ROOT}")
    active_router = router if router is not None else create_default_router()
    return HFSAWebServer((str(host), int(port)), active_router, WEB_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the local HFSA remote-sensing analysis interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind; use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port.")
    parser.add_argument("--open-browser", action="store_true", help="Open the local page after startup.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    server = create_server(args.host, args.port)
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}"
    print("Remote-sensing image-text interpretation interface is ready.")
    print(f"Open: {url}")
    print("Press Ctrl+C to stop.")
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping remote-sensing interface...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
